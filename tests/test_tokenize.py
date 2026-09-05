"""Tokenization tests (issue #3): pluggable Chinese tokenization.

Covers both branches:
- without jieba  -> exact historical per-char behavior
- with jieba     -> word-level Chinese tokens (skipped if jieba not installed)

Note: jieba's stock dictionary splits 机器学习 into the two words 机器|学习
(both multi-char, no more single-char noise). Dictionary-known compounds like
中华人民共和国 stay a single token. Extra domain terms can be forced into
single tokens via AME_JIEBA_WORDS.
"""
import pytest

from engine import bm25, db, tokenize


def _remember(topic, summary):
    from engine.remember import remember
    return remember(topic=topic, summary=summary)


class TestFallbackPerChar:
    """Without jieba, behavior must be exactly the historical per-char split."""

    @pytest.fixture(autouse=True)
    def _no_jieba(self, monkeypatch):
        monkeypatch.setattr(tokenize, "_jieba", None)

    def test_tokens_per_char(self):
        toks = tokenize.tokens("机器学习 awesome")
        assert toks == ["awesome", "机", "器", "学", "习"]

    def test_fts_tokenize_unchanged(self):
        assert db._fts_tokenize("机器学习") == ["机", "器", "学", "习"]

    def test_fts_tokens_no_phrases(self):
        assert tokenize.fts_tokens("机器学习") == ["机", "器", "学", "习"]

    def test_empty_and_none(self):
        assert tokenize.tokens(None) == []
        assert tokenize.tokens("") == []

    def test_bm25_still_matches_per_char(self):
        _remember("word doc", "机器学习加速检索")
        _remember("filler", "unrelated english note")
        hits = bm25.search("机器学习", k=2)
        assert hits and hits[0][1] > 0


@pytest.mark.skipif(not tokenize.jieba_available(), reason="jieba not installed")
class TestWithJieba:
    def test_word_level_no_single_chars(self):
        toks = tokenize.tokens("机器学习")
        # stock dictionary: 机器|学习 - two word-level tokens, zero single chars
        assert "机器" in toks and "学习" in toks
        assert not any(len(t) == 1 for t in toks if tokenize._CJK_CHAR_RE.fullmatch(t))

    def test_dictionary_compound_is_one_token(self):
        toks = tokenize.tokens("中华人民共和国")
        assert "中华人民共和国" in toks

    def test_mixed_ascii_chinese(self):
        toks = tokenize.tokens("vector search 机器学习")
        assert toks[:2] == ["vector", "search"]
        assert "机器" in toks and "学习" in toks

    def test_fts_phrases_emitted(self):
        toks = db._fts_tokenize("机器学习")
        # phrases carry inner spaces: query-side unicode61 would otherwise keep
        # 机器 as one token, which can never match the per-char index
        assert toks == ['"机 器"', '"学 习"']

    def test_fts_index_expansion(self):
        expanded = tokenize.fts_index_text("BM25与vector search的混合检索，机器学习加速")
        # CJK runs become space-separated chars WITH boundaries, so an ASCII
        # word glued to a CJK char no longer merges into one unicode61 token.
        assert "BM25 与 vector" in expanded
        assert "机 器 学 习 加 速" in expanded

    def test_fts_phrase_query_matches(self):
        """The phrase-OR query must be valid FTS5 syntax and return hits."""
        _remember("向量检索", "BM25 与 vector search 的混合检索，机器学习加速")
        _remember("无关记忆", "今天天气不错")
        conn = db.get_conn()
        try:
            assert db.fts5_available(), "FTS5 not available in this build"
            hits = db.fts5_search(conn, "机器学习", k=5)
            assert hits  # matches via phrases "机器"/"学习" or their chars
        finally:
            conn.close()

    def test_bm25_word_outranks_partial_doc(self):
        """A doc containing both query words (机器 AND 学习) must outrank a
        doc containing only one of them - word-level precision, not char noise."""
        _remember("both words", "本项目研究机器学习的方法论")
        _remember("one word", "修理机器的师傅很专业，学生学习习惯不错")
        hits = bm25.search("机器学习", k=2)
        assert hits
        conn = db.get_conn()
        try:
            topics = {r[0]: conn.execute(
                "SELECT topic FROM episodic WHERE rowid=?", (r[0],)
            ).fetchone()["topic"] for r in hits}
        finally:
            conn.close()
        assert topics[hits[0][0]] == "both words"


@pytest.mark.skipif(not tokenize.jieba_available(), reason="jieba not installed")
class TestUserWords:
    """AME_JIEBA_WORDS forces domain terms to stay single tokens."""

    def test_user_word_single_token(self, monkeypatch):
        monkeypatch.setenv("AME_JIEBA_WORDS", "机器学习,深度学习")
        monkeypatch.setattr(tokenize, "_user_words_loaded", False)
        try:
            toks = tokenize.tokens("机器学习入门")
            assert "机器学习" in toks          # now one token, not 机器|学习
        finally:
            # jieba's dictionary is process-global; undo for other tests
            tokenize._jieba.del_word("机器学习")
            tokenize._jieba.del_word("深度学习")
            monkeypatch.delenv("AME_JIEBA_WORDS", raising=False)
