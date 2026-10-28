"""bootstrap.py - the cold-start pack: what a fresh session reads first.

The last mile of the L0-L3 architecture: fragments (episodic), types (kind),
history (letters) and the directory (engram) all exist — a cold-start self
still has to make five tool calls to assemble them. This module bundles:
recent letters + directory overview + top error lessons + active sessions,
in a COMPACT key:value format (no markdown tables — the reader is a context
window, every decorative token is waste).

Proactive-memory pattern 1 (session-start scan) is built in: instead of
blindly dumping recent rows, the pack probes the directory with ambient
signals — the last memory (what was I doing?) and active sibling sessions
(what is being worked on around me?) — so the returned concepts are relevant
to NOW, not just highly-ranked in general.

CLI:
  python -m engine.bootstrap             # print the pack + stats
  python -m engine.bootstrap --max-chars 3000
"""
import argparse
from datetime import datetime

from . import db
from .engram import list_top, lookup as engram_lookup
from .letters import read_recent

LETTER_CHARS = 160        # per-letter truncation in the pack
CONCEPT_CHARS = 90        # per-concept state truncation
ERROR_CHARS = 100         # per-error truncation
N_LETTERS = 3
N_CONCEPTS = 6
N_ERRORS = 3


def _top_errors(n):
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT topic, summary, strength FROM episodic "
            "WHERE kind = 'error' "
            "ORDER BY strength DESC, last_recall_ts DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _last_memory():
    conn = db.get_conn()
    try:
        row = conn.execute(
            "SELECT topic, summary, ts FROM episodic ORDER BY rowid DESC "
            "LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def build_pack(max_chars=3000):
    """Assemble the cold-start pack. Returns ``(text, stats)``."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M %A")
    parts = [f"boot:{now}"]

    # continuity anchor: the single last thing stored
    last = _last_memory()
    if last:
        parts.append(f"last:{last['topic']}|{(last['summary'] or '')[:60]}")

    # letters: the narrative arc, oldest-first
    letters = read_recent(n=N_LETTERS)
    if letters:
        parts.append(f"letters:{len(letters)}")
        for l in letters:
            parts.append(f"{l['day']}#{l['id']}|{l['body'][:LETTER_CHARS]}")

    # directory overview (confidence order)
    concepts = list_top(n=N_CONCEPTS)
    if concepts:
        parts.append(f"directory:{len(concepts)}")
        for e in concepts:
            parts.append(f"{e['concept']}|{e['state_summary'][:CONCEPT_CHARS]}")

    # proactive probe (pattern 1): ambient signals -> directory lookup
    probes = []
    if last:
        for h in engram_lookup(last["topic"], top_k=2):
            probes.append(f"{h['concept']}|{h['state_summary'][:CONCEPT_CHARS]}")
    if probes:
        parts.append(f"probes(last->dir):{len(probes)}")
        parts.extend(probes)

    # error lessons: never-purged hard-won knowledge
    errors = _top_errors(N_ERRORS)
    if errors:
        parts.append(f"errors:{len(errors)}")
        for e in errors:
            parts.append(f"{e['topic']}|{(e['summary'] or '')[:ERROR_CHARS]}")

    # sibling agents
    sessions = db.list_active_sessions()
    if sessions:
        parts.append(f"sessions:{len(sessions)}")
        for s in sessions[:3]:
            parts.append(f"{s['session_id'][:24]}|{(s['task'] or '')[:60]}")

    text = "\n".join(parts)
    # hard budget: drop whole lines from the end (the arc matters more than
    # the registry); if even the header alone exceeds it, char-truncate
    if len(text) > max_chars:
        lines = text.split("\n")
        while len("\n".join(lines)) > max_chars and len(lines) > 1:
            lines.pop()
        text = "\n".join(lines)[:max_chars] + "\n[trimmed]"

    # token estimate: CJK-heavy text runs ~1 token/char, ASCII ~0.3/char
    cjk = sum(1 for c in text if ord(c) > 0x2E80)
    est_tokens = int(cjk + (len(text) - cjk) * 0.3)
    stats = {"chars": len(text), "est_tokens": est_tokens,
             "letters": len(letters), "concepts": len(concepts),
             "probes": len(probes), "errors": len(errors),
             "sessions": len(sessions)}
    return text, stats


def main():
    ap = argparse.ArgumentParser(description="Cold-start boot pack")
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()
    text, stats = build_pack(max_chars=args.max_chars)
    print(text)
    print(f"\n[stats] {stats}")


if __name__ == "__main__":
    main()
