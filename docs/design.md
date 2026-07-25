# Design: Two-Stage Retrieval + Gating

This document explains *why* the engine doesn't just do plain top-k semantic
search, and what each knob in `.env` actually controls.

## The problem: context pollution

Pure semantic recall (Mem0, Chroma, naive RAG) pulls the k nearest neighbors by
embedding distance and stuffs them all into the prompt. Two failure modes:

1. **Semantic-adjacent noise**: "How do I optimize a momentum factor?" recalls
   the memory "discussed the weather" because both are short sentences and
   embeddings have a nonzero baseline similarity. It's *close-ish* but useless.
2. **No notion of importance**: a 6-month-old one-liner ranks the same as the
   thing you discussed yesterday, as long as the cosine is similar.

Stuffing this junk into the system prompt burns context tokens and misleads the
LLM. We want **wide recall, strict output**.

## The pipeline

```
query ──► embed ──► [ Stage A: KNN pool=15 ] ──► 15 candidates
                         │
                         ▼
                   [ Gate: distance > THRESHOLD? drop ]
                         │
                         ▼
                   [ Rerank: score = α·strength + (1-α)·sim ]
                         │
                         ▼
                   top-k survivors ──► prompt
```

### Stage A: wide recall (`AME_RECALL_POOL=15`)

Pull a generously-sized candidate set with KNN. Cheap, over-inclusive on
purpose. If the pool is smaller than `top_k`, we just take what's there.

### Gate: kill pure noise (`AME_RECALL_THRESHOLD=0.9`)

Drop anything whose distance exceeds the threshold. This is intentionally
**lenient** - we only kill things that aren't even *close*. The interesting
filtering happens in rerank, not here.

Why not a strict gate? Because distance alone is a poor relevance signal - the
rerank step combines it with importance.

### Rerank: relevance + importance (`AME_ALPHA=0.5`)

```
score = α · strength_now + (1 - α) · sim
```

where:
- `sim = 1 - distance`  (similarity from the KNN distance)
- `strength_now = exp(-Δt / τ)`  (real-time Ebbinghaus decay, see below)
- `α ∈ [0, 1]` trades off **exploit** (strong memories, α→1) vs **explore**
  (raw similarity, α→0). `0.5` is the safe default.

This is the key idea: **similarity gets you in the door, importance decides the
ranking.** A memory that is both topically relevant AND frequently recalled
wins over a slightly-more-similar but never-used one.

## Ebbinghaus decay as lightweight utility

`strength_now = exp(-Δt / τ)` with:
- `Δt` = days since last recall (or creation)
- `τ` = time constant; **`τ *= 1.5` on every recall**, so well-worn memories
  decay slower.

This is our substitute for the **Q-value** in the SJTU [MemRL paper]. We borrow
the "two-stage retrieval + value-aware rerank" structure but drop the RL part.

### Why not full RL?

MemRL trains a Q-network over memories using task reward. In a **dialogue**
agent there is no clean reward signal:

- A coding task has pass/fail - usable reward.
- A conversation has no ground truth. "Was this recalled memory helpful?" is
  unobservable. Fabricating a reward proxy injects more noise than signal into
  Q, and you end up with an unstable value function.

Ebbinghaus decay gives us a **free, monotonic** utility proxy: frequently
recalled → important → decays slowly. No training, no reward engineering, no
instability. It is wrong in theory (a memory can be recalled often yet be
useless) but right in practice for the long tail of personal agent memory.

[MemRL paper]: https://arxiv.org/abs/2601.03192

## Storage-time features

### LLM summarization (optional)

When `AME_LLM_*` is configured and `summarize=True`, the engine condenses the
raw turn into one semantic sentence *before* embedding. The sentence is what
gets embedded and retrieved; the raw text is preserved in `raw`.

This matters because raw dialogue is full of structure ("user said... / assistant
replied...") that pollutes embedding space. A clean semantic sentence retrieves
far better. Disabled by default to keep the engine a pure, dependency-free
retrieval system.

### Dedup-merge (`AME_DEDUP_THRESHOLD=0.45`)

On store, the new summary's nearest neighbor is checked. If
`distance ≤ threshold`, we **merge** into the existing record instead of
creating a duplicate:

- refresh `ts` to now
- reset `strength = 1.0`, `τ *= 1.5` (the topic came back - reinforce it)
- append the new `raw` to the existing one
- replace the vector with the new summary's

This keeps the DB from bloating with near-duplicate episodes of recurring topics.

## Tuning cheatsheet

| Symptom | Knob | Direction |
|---|---|---|
| Too many irrelevant memories in prompt | `AME_RECALL_THRESHOLD` | ↓ (stricter) |
| Important stuff not surfacing | `AME_ALPHA` | ↑ (weight strength more) |
| Old memories cluttering results | run `forget --purge`, or ↓ `AME_MIN_STRENGTH` |
| Same topic stored as many near-dupes | `AME_DEDUP_THRESHOLD` | ↑ (merge more aggressively) |
| Recall feels slow at scale | `AME_RECALL_POOL` | ↓ (narrower candidate set) |
