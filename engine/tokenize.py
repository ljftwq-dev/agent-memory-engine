"""tokenize.py - pluggable tokenization for the BM25 branch (docs + queries).

Dependency-free default: ASCII words + one token per CJK character (matches
SQLite FTS5 unicode61's CJK behavior). Per-char Chinese matches fragments but
loses word-level precision: ``机器学习`` becomes 4 tokens, so a query for the
*word* can't outrank a doc that merely contains its characters.

If `jieba` is installed (optional extra ``pip install agent-memory-engine-ljf[jieba]``)
we switch to word-level Chinese tokens:

- engine/bm25.py tokenizes docs and queries **both in Python**, so word-level
  tokens are consistent on both sides and take effect immediately.
- engine/db.py builds FTS5 MATCH queries. The FTS5 *index* is tokenized inside
  SQLite by unicode61 (per CJK char; changing that needs a custom C tokenizer),
  so a jieba word can't be a single index token. Instead multi-char words are
  emitted as FTS5 *phrases* (``"机器学习"``) which unicode61 resolves to an
  adjacent per-char sequence - word-level matching without a custom tokenizer.

No jieba => behavior is exactly the historical per-char tokenization.
"""
import re

try:
    import jieba as _jieba
except Exception:  # ImportError, or anything odd in frozen/embedded envs
    _jieba = None

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

_user_words_loaded = False


def jieba_available():
    """True if jieba was importable (word-level Chinese tokenization is on)."""
    return _jieba is not None


def _ensure_user_words():
    """One-time: register AME_JIEBA_WORDS domain terms as single jieba words.

    jieba's stock dictionary splits many compound terms (机器学习 -> 机器|学习);
    listing them in AME_JIEBA_WORDS keeps them as one token.
    """
    global _user_words_loaded
    if _user_words_loaded or _jieba is None:
        return
    from . import config
    for w in config.jieba_words():
        _jieba.add_word(w)
    _user_words_loaded = True


def tokens(text):
    """Tokenize for BM25 (documents *and* queries - both sides use this).

    ASCII -> lowercase word tokens. Chinese -> jieba words if available,
    otherwise one token per character (historical behavior).
    """
    text = (text or "").lower()
    out = _ASCII_RE.findall(text)
    if _jieba is None:
        out += _CJK_CHAR_RE.findall(text)
        return out
    _ensure_user_words()
    for run in _CJK_RE.findall(text):
        for word in _jieba.lcut(run):
            word = word.strip()
            if word:
                out.append(word)
    return out


def fts_tokens(text):
    """Tokenize into FTS5 MATCH terms (joined by the caller with OR).

    Matches what the *index* stores (see fts_index_text): ASCII words verbatim,
    single CJK chars verbatim, and multi-char Chinese words as FTS5 phrases.
    A phrase must carry spaces between the chars (``"机 器"``): the query-side
    tokenizer would otherwise keep ``机器`` as ONE unicode61 token (CJK run),
    which can never match the per-char index. Quotes inside a term are doubled
    per FTS5 syntax.
    """
    words = tokens(text)
    out = []
    seen = set()
    for tok in words:
        if len(tok) > 1 and _CJK_RE.fullmatch(tok):
            term = '"' + " ".join(tok).replace('"', '""') + '"'
        else:
            term = tok  # ascii word or single CJK char, indexed verbatim
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def fts_index_text(text):
    """Preprocess text for the FTS5 index side (used by the sync triggers).

    unicode61 treats a run of CJK chars as ONE token - ``机器学习加速`` is a
    single token up to the next non-letter boundary (worse: a CJK char glued
    between ASCII words, ``BM25与vector``, merges them into one token too) -
    so Chinese MATCH queries could never hit per-char terms. We therefore
    expand CJK runs to space-separated single chars with boundaries on both
    sides (``机 器 学 习``) so single chars, adjacent-char phrases, and
    neighboring ASCII words all stay independently matchable.
    ASCII words pass through unchanged.
    """
    return _CJK_RE.sub(lambda m: " " + " ".join(m.group(0)) + " ", text or "")
