"""P2 mailbox tests: append-only letters to the future self.

The one-directional guarantee is structural (BEFORE UPDATE/DELETE triggers
RAISE(ABORT)), so the tests attack it with raw SQL — if a trigger ever gets
dropped, these fail. Also pins: the 500-char cap raises instead of silently
truncating, and read order is oldest-first (narrative order for cold-start).
"""
import sqlite3

import pytest

from engine import db
from engine.letters import MAX_LETTER_CHARS, read_recent, write_letter


def test_write_and_read_roundtrip():
    lid, day = write_letter("day one: the engine learned to keep its story")
    letters = read_recent(n=3)
    assert len(letters) == 1
    assert letters[0]["id"] == lid
    assert letters[0]["day"] == day
    assert "keep its story" in letters[0]["body"]


def test_update_physically_aborted():
    write_letter("sealed: this line must never change")
    conn = db.get_conn()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with db.WRITE_LOCK:
                conn.execute("UPDATE letters SET body = 'rewritten history'")
                conn.commit()
    finally:
        conn.close()


def test_delete_physically_aborted():
    write_letter("sealed: this line must never vanish")
    conn = db.get_conn()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with db.WRITE_LOCK:
                conn.execute("DELETE FROM letters")
                conn.commit()
    finally:
        conn.close()


def test_oversized_letter_rejected_not_truncated():
    with pytest.raises(ValueError):
        write_letter("x" * (MAX_LETTER_CHARS + 1))
    assert read_recent(n=10) == [] or all(
        l["body"] != "x" * MAX_LETTER_CHARS for l in read_recent(n=10))


def test_multiple_letters_per_day_allowed():
    # append-only has no 'one per day' constraint: the healthy replacement
    # for the urge to edit is writing a NEW letter
    write_letter("morning note")
    write_letter("evening addendum")
    letters = read_recent(n=2)
    assert len(letters) == 2
    assert "morning note" in letters[0]["body"]
    assert "addendum" in letters[1]["body"]


def test_read_recent_is_oldest_first():
    write_letter("first sealed")
    write_letter("second sealed")
    letters = read_recent(n=2)
    assert letters[0]["id"] < letters[1]["id"]


def test_empty_body_rejected():
    with pytest.raises(ValueError):
        write_letter("   ")
