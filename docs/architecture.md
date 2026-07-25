# Architecture

## Four-layer memory model

The engine implements the **episodic** layer of a classic four-layer agent
memory model. The other three layers live outside this repo (in your agent's
host), but the design is meant to slot in cleanly:

| Layer | What | Where in this engine |
|---|---|---|
| **Working memory** | The current turn's context window | Not stored - that's the agent host's job |
| **Episodic memory** | "What happened in past sessions" | **`episodic` table - this engine's core** |
| **Semantic memory** | Structured facts / knowledge base | Out of scope; point your agent at any KB (Obsidian, Notion, a vector DB...) |
| **Procedural memory** | Skills / how-to (your `skills/` files) | Out of scope; the agent's filesystem |

The engine only owns episodic memory on purpose - one job, done well.

## Storage layout

A single SQLite database file holds everything:

```
memory.db
├── episodic          (regular table)   metadata: topic / summary / raw /
│                                       strength / tau / last_recall_ts / created_ts
└── episodic_vec      (vec0 virtual)    FLOAT[dim] embedding, joined by rowid
```

`sqlite-vec` provides the `vec0` virtual table and KNN via
`WHERE embedding MATCH ? AND k = N`. No separate vector server, no extra
process - copy the `.db` file and you have the whole memory.

## Module map

```
config.py     reads .env / env vars; every knob lives here (no hardcoded paths)
db.py         connection + schema init; vec0 table; dim detection
embed.py      singleton embedder (BGE-m3 by default); deterministic hash fallback
recall.py     two-stage retrieval + gating + Ebbinghaus strength     [the core]
remember.py   store with optional LLM summary + automatic dedup-merge
forget.py     nightly decay loop; optional purge of weak memories
server.py     stdlib HTTP server exposing recall/remember/recent/search/forget
```

## Data flow (one turn)

```
user message
   │
   ▼
agent plugin ──► GET /recall?q=<user msg>  ──► engine
                                                  │  Stage A: KNN pool=15
                                                  │  Gate   : drop distance > threshold
                                                  │  Stage B: score, rerank, top-k
                                                  ▼
agent plugin ◄── returns related memories
   │
   │ injects memories into system prompt
   ▼
LLM answers
   │
   ▼
session idle ──► POST /remember {topic, summary}  ──► engine stores (dedup or create)
                                                        optional: LLM summarizes first
```

## Why SQLite + sqlite-vec

- **Zero ops**: one file, no server process (vs Chroma / Qdrant / Milvus).
- **Good enough at this scale**: episodic memory for a personal agent is
  hundreds to low-thousands of rows, not millions. SQLite handles that trivially.
- **Portable**: the entire memory is a single copyable `.db` file.
- **Upgrade path**: if you outgrow it, swap `db.py` for a Postgres+pgvector
  backend without touching `recall.py` / `remember.py`.
