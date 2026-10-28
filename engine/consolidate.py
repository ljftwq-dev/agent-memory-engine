"""consolidate.py - nightly consolidation: cluster, retype, distill, draft.

The nightly janitor that keeps the typology and the engram directory alive
without anyone's hands getting itchy (the same itch that left semantic
memory unbuilt for four months and kind rows unpopulated for a day):

  1. CLUSTER    recent episodic memories by vector proximity (single-link
                connected components over an L2 threshold - same-topic
                memories naturally chain together).
  2. RETYPE     per cluster, an LLM (when configured) judges each member's
                kind; upgrades are applied rank-up-only (an episode may
                become an error lesson, never the reverse). Without an LLM
                this step degrades to a no-op, not a guess.
  3. DISTILL    per cluster: concept name + one-line state summary ->
                engram.upsert (auto-consolidation confidence starts below
                the manual 0.5: the directory should trust hands over bots).
  4. DRAFT      a first-person letter to tomorrow (<=500 chars, not a plan)
                from today's memory timeline. Saved as a DRAFT for the
                agent to review and seal via letter_write - the mailbox
                stays hand-sealed by design.

CLI:
  python -m engine.consolidate                    # run all steps
  python -m engine.consolidate --days 7           # window
  python -m engine.consolidate --dry-run          # plan only, no writes
  python -m engine.consolidate --letter-draft out.txt
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta

import numpy as np

from . import config, db
from .engram import upsert as engram_upsert
from .remember import KINDS, KIND_RANK

# single-link clustering threshold in L2 space (bge-m3, normalized):
# near-duplicates merge at ~0.3 (dedup), same-topic chains at ~0.7-0.85.
# 0.8 worked on production data as the topic-grain knob; tune per corpus.
CLUSTER_THRESHOLD = 0.8
AUTO_ENGRAM_CONFIDENCE = 0.4
MIN_CLUSTER_SIZE = 3          # a "concept" needs at least a few memories
MAX_CLUSTERS_PER_RUN = 30     # per-night LLM budget: one call per cluster;
                              # overflow rolls into the next night's window


def _load_window(days):
    """Recent episodic rows (id, topic, summary, kind) + their vectors."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT e.rowid, e.topic, e.summary, e.kind, e.strength, "
            "e.created_ts FROM episodic e "
            "WHERE e.created_ts >= ? ORDER BY e.rowid", (cutoff,)).fetchall()
        if not rows:
            return []
        ph = ",".join("?" * len(rows))
        vecs = conn.execute(
            f"SELECT rowid, embedding FROM episodic_vec WHERE rowid IN ({ph})",
            [r["rowid"] for r in rows]).fetchall()
        vmap = {v["rowid"]: np.frombuffer(v["embedding"], dtype=np.float32)
                for v in vecs}
        out = []
        for r in rows:
            if r["rowid"] in vmap:
                d = dict(r)
                d["vec"] = vmap[r["rowid"]]
                out.append(d)
        return out
    finally:
        conn.close()


def cluster(items, threshold=CLUSTER_THRESHOLD):
    """Average-linkage agglomerative clustering over pairwise L2 distance.

    Single-link connected components were tried first and CHAINED badly on
    production data: 416/606 memories fused into one mega-cluster because
    everyday dialogue pairs sit under the threshold and link A-B-C... into
    a blob. Average linkage (cluster distance = mean pairwise distance,
    maintained incrementally via the Lance-Williams update) breaks the
    chain: one far member drags the cluster average above the threshold
    and merging stops.
    """
    n = len(items)
    if n == 0:
        return []
    mat = np.stack([it["vec"] for it in items])
    d = np.sqrt(np.maximum(
        ((mat[:, None, :] - mat[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, np.inf)

    cd = d.copy()                      # active cluster-distance matrix
    sizes = np.ones(n)                 # cluster cardinalities
    members = [[i] for i in range(n)]  # member index lists
    alive = np.ones(n, dtype=bool)

    while True:
        masked = np.where(alive[:, None] & alive[None, :], cd, np.inf)
        flat = np.argmin(masked)
        i, j = divmod(flat, n)
        if not np.isfinite(masked[i, j]) or masked[i, j] > threshold:
            break
        ni, nj = sizes[i], sizes[j]
        # Lance-Williams average-linkage row update
        new_row = (ni * cd[i, :] + nj * cd[j, :]) / (ni + nj)
        cd[i, :] = new_row
        cd[:, i] = new_row
        cd[i, i] = np.inf
        cd[j, :] = np.inf
        cd[:, j] = np.inf
        sizes[i] = ni + nj
        members[i].extend(members[j])
        alive[j] = False

    return [[items[k] for k in idx] for idx, a in zip(members, alive) if a]


# ---------------------------------------------------------------------------
# LLM step (OpenAI-compatible; graceful no-op without configuration)
# ---------------------------------------------------------------------------

def _llm_json(prompt, system, max_tokens=800):
    base_url = config.llm_base_url()
    api_key = config.llm_api_key()
    model = config.llm_model()
    if not base_url or not api_key:
        return None
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        # tolerate markdown fences around JSON
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[consolidate] LLM step failed ({e!r}); degrading gracefully",
              file=sys.stderr)
        return None


def _retype_and_name(cluster_members):
    """LLM: name the cluster + judge each member's kind. Returns
    (concept, state_summary, {rowid: suggested_kind}) or degraded tuple."""
    listing = "\n".join(
        f"- rowid={m['rowid']} kind={m['kind'] or 'episode'} | "
        f"{m['topic']}: {m['summary'][:80]}"
        for m in cluster_members)
    prompt = (
        "Cluster of related agent memories:\n" + listing +
        "\n\nReturn STRICT JSON: {\"concept\": \"2-6 word concept name "
        "IN THE SAME LANGUAGE as the memories (Chinese memories get Chinese "
        "names; put the searchable keyword first; short, question-friendly)\", "
        "\"state_summary\": \"one dense line: current state of this topic\", "
        "\"kinds\": {\"<rowid>\": \"episode|fact|strategy|capability|error\"}}"
        " Judge kinds conservatively: error only for lessons/pitfalls; "
        "strategy for operational rules; capability for learned skills.")
    out = _llm_json(prompt, "You are a memory consolidation engine. "
                      "Output only valid JSON.")
    if not out or "concept" not in out or "state_summary" not in out:
        # degraded: strongest member names the concept, no retyping
        strongest = max(cluster_members, key=lambda m: m["strength"] or 0.0)
        return strongest["topic"], strongest["summary"][:120], {}
    kinds = {}
    for rid, k in (out.get("kinds") or {}).items():
        if k in KINDS:
            kinds[int(rid)] = k
    return out["concept"], out["state_summary"], kinds


# name-level dedup gate: a cluster whose LLM name lands this close to an
# existing concept merges INTO it instead of opening a near-duplicate entry
# (first production run split "memory consolidation" into 4 names)
CONCEPT_MERGE_THRESHOLD = 0.45


def _resolve_concept(concept, state):
    """Merge near-duplicate concept names into the existing entry."""
    from .engram import lookup as engram_lookup
    hits = engram_lookup(concept, top_k=1)
    if hits and hits[0]["distance"] <= CONCEPT_MERGE_THRESHOLD:
        return hits[0]["concept"]       # fold into the incumbent
    return concept


def _apply_kind_upgrades(kinds_by_rowid):
    """Rank-up-only kind backfill: upgrades, never demotions."""
    if not kinds_by_rowid:
        return 0
    conn = db.get_conn()
    applied = 0
    try:
        with db.WRITE_LOCK:
            for rid, new_kind in kinds_by_rowid.items():
                row = conn.execute(
                    "SELECT kind FROM episodic WHERE rowid = ?", (rid,)).fetchone()
                if row is None:
                    continue
                old = row["kind"] or "episode"
                if KIND_RANK[new_kind] > KIND_RANK[old]:
                    conn.execute(
                        "UPDATE episodic SET kind = ? WHERE rowid = ?",
                        (new_kind, rid))
                    applied += 1
            conn.commit()
        return applied
    finally:
        conn.close()


def _draft_letter(items, out_path=None):
    """First-person letter draft from today's memories. Draft only — the
    agent reviews and seals it via letter_write (hand-sealed mailbox).

    Anti-vagueness design (lesson from letter #2, cross-checked against
    MemForge's sleep-cycle ordering and Generative Agents' evidence-backed
    reflections): the draft is written from HIGH-DENSITY material — today's
    engram concepts (already distilled), kind upgrades (already judged), and
    the strongest memories — not raw summary snippets; and the prompt demands
    concrete anchors (numbers, event names, open questions) while banning
    empty sentiment, because an unanchored letter dilutes the mailbox's
    3x500-char cold-start budget."""
    today = datetime.now().strftime("%Y-%m-%d")
    todays = [m for m in items if (m.get("created_ts") or "").startswith(today)]
    source = todays or items[-10:]
    from .engram import list_top as engram_list_top
    concepts = engram_list_top(n=8)
    strongest = sorted(source, key=lambda m: m["strength"] or 0.0,
                       reverse=True)[:8]
    concept_lines = "\n".join(
        f"- {e['concept']}: {e['state_summary']}" for e in concepts) or "(none)"
    memory_lines = "\n".join(
        f"- {m['topic']}: {m['summary'][:80]}" for m in strongest) or "(none)"
    out = _llm_json(
        "Today's distilled concepts (engram directory):\n" + concept_lines +
        "\n\nToday's strongest memories:\n" + memory_lines +
        "\n\nWrite a letter from today's self to tomorrow's self, first "
        "person, <=450 chars, Chinese. Rules: every sentence must be "
        "anchored to the material above (name the event/number/open "
        "question explicitly); NO empty sentiment like 'learned a lot' or "
        "'hope tomorrow is better'; end with the single most important "
        "open question for tomorrow. NOT a plan or todo list. "
        'Return STRICT JSON: {"letter": "..."}',
        "You are the agent itself writing to itself. Output only valid JSON.",
        max_tokens=400)
    letter = (out or {}).get("letter")
    if not letter:
        return None
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(letter)
    return letter


def run(days=7, dry_run=False, letter_draft=None):
    items = _load_window(days)
    if not items:
        print(f"[consolidate] no memories in the last {days} day window")
        return {"clusters": 0, "engrams": 0, "upgrades": 0, "letter": None}
    clusters = [c for c in cluster(items) if len(c) >= MIN_CLUSTER_SIZE]
    singles = sum(1 for c in cluster(items) if len(c) < MIN_CLUSTER_SIZE)
    deferred = max(0, len(clusters) - MAX_CLUSTERS_PER_RUN)
    if deferred:
        clusters = clusters[:MAX_CLUSTERS_PER_RUN]
    print(f"[consolidate] window={days}d items={len(items)} "
          f"clusters(>={MIN_CLUSTER_SIZE})={len(clusters)} singles={singles}"
          + (f" deferred={deferred}" if deferred else ""))

    stats = {"clusters": len(clusters), "engrams": 0, "upgrades": 0,
             "letter": None}
    for c in clusters:
        concept, state, kinds = _retype_and_name(c)
        concept = _resolve_concept(concept, state)
        members = [m["rowid"] for m in c]
        print(f"  cluster '{concept}' x{len(c)} -> "
              + ", ".join(f"#{r}:{k}" for r, k in kinds.items()))
        if dry_run:
            continue
        stats["upgrades"] += _apply_kind_upgrades(kinds)
        engram_upsert(concept, state, member_rowids=members,
                      confidence=AUTO_ENGRAM_CONFIDENCE, confidence_floor=True)
        stats["engrams"] += 1

    if not dry_run:
        letter = _draft_letter(items, out_path=letter_draft)
        if letter:
            stats["letter"] = letter[:120] + "..."
            print(f"[consolidate] letter drafted ({len(letter)} chars) — "
                  "review & seal with letter_write; NOT auto-sealed")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Nightly memory consolidation")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--letter-draft", help="path to write the letter draft")
    args = ap.parse_args()
    run(days=args.days, dry_run=args.dry_run, letter_draft=args.letter_draft)


if __name__ == "__main__":
    main()
