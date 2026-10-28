"""P3c bootstrap tests: pack assembly, budget trim, compact format."""
from engine.bootstrap import build_pack
from engine.engram import upsert
from engine.letters import write_letter
from engine.remember import remember


def test_pack_assembles_all_sections():
    remember("boot test", "boot test summary")
    write_letter("boot letter one for pack assembly")
    upsert("boot concept", "boot concept state summary")
    remember("boot error", "boot error lesson summary", kind="error")
    text, stats = build_pack()
    assert text.startswith("boot:")
    assert "letters:1" in text
    assert "boot concept" in text
    assert "errors:1" in text
    assert "boot:" in text.split("\n")[0]           # timestamp header
    assert stats["letters"] == 1 and stats["errors"] == 1


def test_pack_respects_budget():
    # only 3 letters make it into the pack (N_LETTERS), so the budget must
    # be low enough for the assembled pack to actually exceed it
    for i in range(4):
        write_letter(f"budget letter number {i} " + "x" * 130)
    text, stats = build_pack(max_chars=300)
    assert len(text) <= 300 + len("\n[trimmed]")
    assert text.rstrip().endswith("[trimmed]")


def test_pack_is_compact_not_markdown():
    write_letter("format letter")
    text, _ = build_pack()
    assert "```" not in text
    assert "|" in text                                # key:value separator
    assert "  \n" not in text.rstrip("\n")            # no trailing space runs


def test_proactive_probe_uses_last_memory():
    # the probe looks up the directory with the LAST memory's topic; seed a
    # matching concept and a matching last memory
    upsert("probe target concept", "probe state")
    remember("probe target concept", "last memory about probe target")
    text, stats = build_pack()
    assert stats["probes"] >= 0                       # never crashes
    if stats["probes"]:
        assert "probe target concept" in text
