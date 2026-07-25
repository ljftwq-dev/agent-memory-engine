"""Tests for the multi-agent session registry (collaboration layer).

Each running agent registers its current task so siblings can see what the
others are doing - the "shared working state" layer on top of episodic memory.
"""
from datetime import datetime, timedelta

from engine import db


def test_register_and_list():
    db.register_session("s1", "building memory engine")
    active = db.list_active_sessions()
    assert len(active) == 1
    assert active[0]["session_id"] == "s1"
    assert active[0]["task"] == "building memory engine"
    assert active[0]["status"] == "active"


def test_register_multiple():
    db.register_session("s1", "task A")
    db.register_session("s2", "task B")
    active = db.list_active_sessions()
    assert len(active) == 2
    assert {s["task"] for s in active} == {"task A", "task B"}


def test_reregister_reactivates_and_updates_task():
    db.register_session("s1", "old task")
    db.finish_session("s1")
    assert db.list_active_sessions() == []
    db.register_session("s1", "new task")  # re-register re-activates
    active = db.list_active_sessions()
    assert len(active) == 1
    assert active[0]["task"] == "new task"
    assert active[0]["status"] == "active"


def test_heartbeat_updates_progress():
    db.register_session("s1", "task")
    ok = db.heartbeat_session("s1", progress="halfway done")
    assert ok is True
    active = db.list_active_sessions()
    assert active[0]["progress"] == "halfway done"


def test_heartbeat_unknown_session():
    assert db.heartbeat_session("nonexistent", progress="x") is False


def test_finish_removes_from_active():
    db.register_session("s1", "task")
    assert len(db.list_active_sessions()) == 1
    assert db.finish_session("s1", result="done") is True
    assert db.list_active_sessions() == []


def test_finish_unknown_session():
    assert db.finish_session("nonexistent") is False


def test_stale_session_excluded_by_timeout():
    """Sessions silent longer than the timeout are excluded from active list."""
    db.register_session("s1", "fresh")
    db.register_session("s2", "stale")
    old = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
    conn = db.get_conn()
    try:
        conn.execute("UPDATE session SET updated_ts = ? WHERE session_id = ?",
                     (old, "s2"))
        conn.commit()
    finally:
        conn.close()
    # default timeout = 2h: only s1 (fresh) shows up
    active = db.list_active_sessions()
    assert {s["session_id"] for s in active} == {"s1"}
