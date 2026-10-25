"""tests/test_social.py - social memory layer (fan cards + mood state)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import db
from engine import social


@pytest.fixture()
def conn(tmp_path):
    c = db.get_conn(str(tmp_path / "social.db"))
    social.init_social(c)
    yield c
    c.close()


def test_fan_lifecycle(conn):
    card = social.upsert_fan("bilibili", "10086", nickname="墨迹一号",
                             impression="第一天来评论区打卡", conn=conn)
    assert card["nickname"] == "墨迹一号"
    assert card["favor"] == 50
    assert card["relation"] == "fan"

    got = social.get_fan("bilibili", "10086", conn=conn)
    assert got["relation"] == "fan"

    card = social.upsert_fan("bilibili", "10086", favor_delta=45,
                             impression="连投三硬币", conn=conn)
    assert card["favor"] == 95
    assert card["relation"] == "friend"
    assert "连投三硬币" in card["impressions"]


def test_favor_clamp_and_relation_floors(conn):
    social.upsert_fan("bilibili", "1", favor=98, conn=conn)
    card = social.bump_favor("bilibili", "1", -200, "巨幅扣分", conn=conn)
    assert card["favor"] == 0
    assert card["relation"] == "stranger"

    card = social.bump_favor("bilibili", "1", 999, "拉满", conn=conn)
    assert card["favor"] == 100
    assert card["relation"] == "friend"


def test_blacklist_override(conn):
    social.upsert_fan("bilibili", "2", favor=99, conn=conn)
    card = social.upsert_fan("bilibili", "2", blacklist=True, conn=conn)
    assert card["relation"] == "blocked"
    card = social.upsert_fan("bilibili", "2", favor_delta=0, conn=conn)
    assert card["relation"] == "blocked", "auto-refresh must not unblock"

    card = social.upsert_fan("bilibili", "2", blacklist=False, conn=conn)
    assert card["relation"] == "friend"


def test_master_relation_manual(conn):
    card = social.upsert_fan("bilibili", "3", favor=10,
                             relation="master", conn=conn)
    assert card["relation"] == "master"


def test_impressions_cap(conn):
    for i in range(25):
        social.upsert_fan("bilibili", "4", impression=f"印象{i}",
                          favor_delta=0, conn=conn)
    card = social.get_fan("bilibili", "4", conn=conn)
    lines = card["impressions"].splitlines()
    assert len(lines) == 20
    assert "印象24" in lines[-1]
    assert "印象0" not in card["impressions"]


def test_list_fans_order(conn):
    social.upsert_fan("bilibili", "a", favor=30, conn=conn)
    social.upsert_fan("bilibili", "b", favor=90, conn=conn)
    social.upsert_fan("discord", "c", favor=60, conn=conn)
    fans = social.list_fans(conn=conn)
    assert [f["uid"] for f in fans[:2]] == ["b", "c"]
    only_bili = social.list_fans(platform="bilibili", conn=conn)
    assert {f["uid"] for f in only_bili} == {"a", "b"}


def test_mood_roundtrip(conn):
    assert social.get_mood(conn=conn) is None
    cur = social.set_mood("炸毛", valence=-40, arousal=80,
                          reason="被说不如DeepSeek", conn=conn)
    assert cur["label"] == "炸毛"
    cur = social.set_mood("开心", valence=60, arousal=50,
                          reason="涨粉100", conn=conn)
    assert cur["label"] == "开心"

    events = social.recent_mood_events(conn=conn)
    assert len(events) == 2
    assert events[0]["label"] == "开心"
    assert events[1]["reason"] == "被说不如DeepSeek"


def test_get_unknown_fan(conn):
    assert social.get_fan("bilibili", "ghost", conn=conn) is None
