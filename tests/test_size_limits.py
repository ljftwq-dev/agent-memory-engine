"""P0 hygiene tests: raw size cap, summary hard limit, recall raw stripping.

The remember() dedup-merge path used to append raw text without bound
(old_raw + "\\n\\n---\\n\\n" + new_raw, forever), so a hot memory merged
hundreds of times grew its raw into a token bomb; and HTTP recall returned
the full raw, ready to be dumped into some future pipeline's prompt. These
tests pin the three write/read guards.
"""
from engine import db
from engine.recall import recall
from engine.remember import MAX_RAW_CHARS, MAX_SUMMARY_CHARS, remember


def _long_text(n):
    base = "lorem ipsum dolor sit amet "
    return (base * (n // len(base) + 1))[:n]


def _max_col_len(col):
    conn = db.get_conn()
    try:
        row = conn.execute(f"SELECT MAX(length({col})) FROM episodic").fetchone()
        return row[0]
    finally:
        conn.close()


def test_merge_raw_capped():
    # identical summaries => vector distance 0 => every call merges into one row
    for i in range(5):
        rowid, action = remember("cap test", "identical summary for merge",
                                 raw=_long_text(4000))
    assert action == "merged"
    assert _max_col_len("raw") <= MAX_RAW_CHARS + 100  # cap + truncation marker


def test_first_insert_raw_capped():
    remember("cap test 2", "another summary", raw=_long_text(20000))
    assert _max_col_len("raw") <= MAX_RAW_CHARS + 100


def test_summary_hard_limit():
    remember("summary cap", _long_text(2000), raw="x")
    assert _max_col_len("summary") <= MAX_SUMMARY_CHARS


def test_recall_strips_raw_by_default():
    remember("strip test", "strip summary", raw=_long_text(1000))
    results = recall("strip summary", top_k=3)
    assert results
    assert all("raw" not in r for r in results)


def test_recall_include_raw_opt_in():
    remember("strip test", "strip summary", raw=_long_text(1000))
    results = recall("strip summary", top_k=3, include_raw=True)
    assert results
    assert any(r.get("raw") for r in results)
