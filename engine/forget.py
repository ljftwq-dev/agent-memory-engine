"""forget.py - Ebbinghaus decay loop (recommended as a nightly cron).

For each memory we recompute ``strength = exp(-dt/tau)``:
  - dt   = days since last recall (or creation)
  - tau  = time constant (days); each recall does tau *= 1.5, so frequently
           recalled memories decay slower
  - smaller strength -> closer to "forgotten"

By default only ``strength`` is updated (reversible). Use ``--purge`` to
actually delete memories below the threshold (and their vector rows).

CLI:
  python -m engine.forget                 # decay only
  python -m engine.forget --dry-run       # preview
  python -m engine.forget --purge         # decay + purge weak memories
"""
import argparse
import math
import os
from datetime import datetime

from . import config, db


def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def decay_all(threshold=0.05, purge=False, dry_run=False, min_age_days=0):
    """Walk all memories, recompute strength, optionally purge.

    Returns ``{total, decayed, purged, kept}``.
    """
    if not os.path.exists(config.db_path()):
        return {"total": 0, "decayed": 0, "purged": 0, "kept": 0}

    conn = db.get_conn()
    now = datetime.now()
    try:
        rows = conn.execute(
            "SELECT rowid, ts, topic, summary, strength, tau, "
            "last_recall_ts, created_ts FROM episodic"
        ).fetchall()

        decayed = purged = kept = 0
        for r in rows:
            ref = (_parse_ts(r["last_recall_ts"])
                   or _parse_ts(r["created_ts"])
                   or _parse_ts(r["ts"]))
            if ref is None:
                kept += 1
                continue
            delta_days = (now - ref).total_seconds() / 86400.0
            if delta_days < min_age_days:
                kept += 1
                continue
            tau_days = max(r["tau"] or 7.0, 0.001)
            new_strength = math.exp(-delta_days / tau_days)

            if dry_run:
                tag = "PURGE" if (purge and new_strength < threshold) else "keep "
                print(f"  [{tag}] #{r['rowid']:>4}  s {r['strength']:.3f}->{new_strength:.3f}  "
                      f"tau={tau_days:.1f}d  dt={delta_days:.1f}d  {(r['topic'] or '')[:40]}")
                continue

            if purge and new_strength < threshold:
                conn.execute("DELETE FROM episodic WHERE rowid = ?", (r["rowid"],))
                conn.execute("DELETE FROM episodic_vec WHERE rowid = ?", (r["rowid"],))
                purged += 1
            else:
                conn.execute("UPDATE episodic SET strength = ? WHERE rowid = ?",
                             (new_strength, r["rowid"]))
                decayed += 1

        if not dry_run:
            conn.commit()
        return {"total": len(rows), "decayed": decayed, "purged": purged, "kept": kept}
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ebbinghaus decay loop (nightly)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="purge threshold (default 0.05)")
    ap.add_argument("--purge", action="store_true",
                    help="actually delete low-strength memories (default: decay only)")
    ap.add_argument("--dry-run", action="store_true", help="preview, don't write")
    ap.add_argument("--min-age", dest="min_age_days", type=float, default=0,
                    help="only touch memories older than N days (protect new ones)")
    args = ap.parse_args()

    if args.dry_run:
        print(f"=== DRY RUN (threshold={args.threshold}, purge={args.purge}) ===")
    r = decay_all(threshold=args.threshold, purge=args.purge,
                  dry_run=args.dry_run, min_age_days=args.min_age_days)
    if not args.dry_run:
        print(f"total={r['total']} decayed={r['decayed']} "
              f"purged={r['purged']} kept={r['kept']}")
