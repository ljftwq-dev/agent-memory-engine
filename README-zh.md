# Agent Memory Engine

[![CI](https://github.com/ljftwq-dev/agent-memory-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ljftwq-dev/agent-memory-engine/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-memory-engine-ljf.svg)](https://pypi.org/project/agent-memory-engine-ljf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

English version: [README.md](README.md)

![Agent Memory Engine 架构](docs/images/architecture.png)

_架构总览——完整说明见 [docs/architecture.md](docs/architecture.md)。_

![Dashboard](docs/images/screenshots/dashboard.png)

_Web 仪表盘——召回查询、浏览最近记忆、查看多 agent 实时会话。_

一个为编程 agent（opencode / claude-code / 任何能讲 HTTP 的 agent）设计的**长期记忆引擎**。让 agent 跨重启“记住”过去的会话：每个新回合自动**召回**相关历史，每个完成的回合自动**存入**一条新记忆。

> 灵感来自上交大的 [MemRL 论文](https://arxiv.org/abs/2601.03192)，做了一个工程取舍：借鉴它的**两阶段检索 + 门控**，放弃完整 RL（对话没有干净的奖励信号——见 [design.md](docs/design.md)）。

---

## 为什么不同（对比 Mem0 / Chroma）

大多数记忆层只做**纯语义召回**——最近邻结果直接塞进 prompt。本引擎不一样：

| 特性 | 带来的价值 |
|---|---|
| **两阶段检索 + 门控** | 宽网 KNN 召回（15 条）→ 丢弃纯噪声 → 用 `score = α·strength + (1-α)·sim` 重排 → 取 top-k。**不再把“语义相邻但无用”的垃圾塞进你的 prompt。** |
| **混合召回（vector + BM25）** | 向量 KNN 抓语义匹配 + FTS5 BM25 抓关键词匹配，用 RRF（倒数秩融合）合并。能捞回纯向量路径漏掉的关键词命中。用 `AME_HYBRID_ENABLE=0` 关闭。 |
| **交叉编码器重排（可选）** | 混合融合之后，用一个 cross-encoder（bge-reranker-v2-m3）对 `(query, candidate)` 对重新打分——经典的两阶段 IR 模式（廉价宽召回 → 精确重排）。默认关闭；模型没加载会优雅跳过。用 `AME_RERANKER_ENABLE=1` 开启。 |
| **多 agent 协同** | 各会话注册自己当前的任务，兄弟会话通过 `GET /sessions/active` 看到“谁在干什么”。一个先做完的 agent 能直接接手兄弟会话的未竟工作，省去你写交接文档。 |
| **LLM 摘要** | 可选地在 embedding 之前把每个回合浓缩成一句语义话（比原始对话检索效果更好）。没配 LLM 时自动回退到原文。 |
| **艾宾浩斯衰减** | 经常被召回的记忆衰减更慢（每次召回 `τ *= 1.5`），长期不用的自然淡出。用进废退，不需要 RL 训练。 |
| **单个 SQLite 文件** | 结构化数据 + 向量索引都在一个 `.db` 里。不需要单独的向量服务器、不跑额外进程——直接拷文件即可。 |
| **agent 与 LLM 无关** | 纯 HTTP。默认 embedder 是 BGE-m3（本地、免费）；LLM 摘要支持任何 OpenAI 兼容端点（GLM / OpenAI / Ollama）。 |

---

## 基准测试

在 40 条编程 agent 记忆 + 24 个手工标注查询（分级相关性）上的检索质量消融实验。衰减已中和，数字反映纯检索/重排质量。完整方法见 [`benchmark/README.md`](benchmark/README.md)。

| 配置 | nDCG@5 | Recall@5 |
|---|---|---|
| 纯向量 | 0.842 | 0.875 |
| 混合（vector + BM25，RRF） | 0.868 | 0.896 |
| **混合 + cross-encoder 重排** | **0.927** | **0.979** |

每一步都值回票价：

| 步骤 | nDCG@5 提升 | Recall@5 提升 |
|---|---|---|
| +混合（RRF 融合） | +0.026 | +0.021 |
| +重排器（cross-encoder） | +0.059 | +0.083 |

cross-encoder 是最大的单步收益——联合阅读 `(query, candidate)` 比分开编码更准，正如 IR 理论预测的那样。

_2026-07-28 复现（BGE-m3 + bge-reranker-v2-m3）。自己跑：`python benchmark/run_benchmark.py`。_

---

## 快速开始

**从 PyPI 安装：**
```bash
pip install agent-memory-engine-ljf
```

**从源码安装：**
```bash
git clone https://github.com/ljftwq-dev/agent-memory-engine
cd agent-memory-engine
pip install -e ".[all]"        # 核心 + 真实 embedding + 开发依赖
cp .env.example .env           # 按需调整（默认值开箱即用）

python -m engine.server        # 启动 HTTP 服务（默认端口 :8765）
```

引擎有两种工作模式：

- **没装 embedding 模型** → 自动哈希兜底（确定性、可复现、*无真实语义*）。适合试 API，召回质量不行。想要真实语义就装 `[embed]` 扩展（BGE-m3）。
- **装了 BGE-m3** → 完整的多语言语义 embedding。

任何 agent 都通过 HTTP 和它对话：

```
GET  /health                 服务状态
GET  /recall?q=&k=3          两阶段语义召回（核心）
GET  /recent?k=5             最近 k 条记忆（按时间）
GET  /search?q=              关键词 LIKE 匹配
POST /remember               存一条记忆 {topic, summary, raw?, ...}
POST /forget                 跑一次艾宾浩斯衰减 {purge?, threshold?}
```

想要 LLM 摘要？在 `.env` 里设 `AME_LLM_BASE_URL` + `AME_LLM_API_KEY`（任何 OpenAI 兼容端点）。留空就是一个纯检索引擎——照样完全可用。

灌点演示数据试试：

```bash
python examples/seed_demo.py
python -m engine.recall "vector search" -k 3
```

---

## 仓库结构

```
agent-memory-engine/
├── engine/            核心引擎
│   ├── config.py      .env / 环境变量加载（无硬编码路径）
│   ├── db.py          SQLite + sqlite-vec schema
│   ├── embed.py       BGE-m3 embedder + 哈希兜底
│   ├── recall.py      两阶段检索 + 门控（核心）
│   ├── reranker.py    可选的 cross-encoder 精确重排
│   ├── remember.py    存储 + 可选 LLM 摘要 + 去重合并
│   ├── forget.py      艾宾浩斯衰减循环（夜间 cron）
│   └── server.py      标准库 HTTP 服务
├── examples/
│   ├── seed_demo.py              灌入通用演示数据
│   ├── opencode/memory.ts        opencode 插件参考（注入 + 召回 + 存储）
│   └── claude-code/              .mcp.json + CLAUDE.md 片段（模型驱动）
├── tests/
│   ├── conftest.py               临时 DB + 强制哈希兜底（CI 友好）
│   ├── test_recall.py            门控 / 去重 / 衰减 / 强化测试
│   └── test_reranker.py          重排兜底 / mock 提升测试
└── docs/
    ├── architecture.md           四层记忆模型 + 模块图
    └── design.md                 为什么两阶段、为什么不上完整 RL
```

---

## 设计原理（一段话讲清）

**问题**：纯语义匹配会造成*上下文污染*——“语义相近” ≠ “有用”，于是把相邻的垃圾铲进 prompt。
**解法（宽进严出）**：
1. **门控只杀纯噪声**（distance > threshold）。混合模式下 BM25 命中可绕过门控。
2. **融合 + 重排决定相关性**：混合模式下相关性 = RRF 融合后的 vector + BM25 排名；否则 `sim = 1 - distance`。如果开了 cross-encoder 重排，用它精确的 `(query, candidate)` 分数替换该相关性。最终 `score = α·strength + (1-α)·相关性`，取 top-k。
3. **`strength` 是轻量级 Utility**：艾宾浩斯 `exp(-Δt/τ)`，每次召回 `τ *= 1.5`。常被召回的记忆保持强健。这替代了 MemRL 的 Q 值，且不需要奖励数据。
4. **不上完整 RL**：对话没有干净的奖励信号；编一个代理奖励带来的噪声比 Q 值本身还大。

完整论述：[`docs/design.md`](docs/design.md)。

---

## 配置（`.env`）

| 键 | 默认值 | 含义 |
|---|---|---|
| `AME_DB_PATH` | `~/.agent-memory/memory.db` | 数据库路径 |
| `AME_EMBED_MODEL` | `BAAI/bge-m3` | embedding 模型（本地免费） |
| `AME_EMBED_DIM` | `1024` | 向量维度（需与模型匹配） |
| `AME_LLM_BASE_URL` | （空 = 关闭） | OpenAI 兼容 API base |
| `AME_LLM_API_KEY` | （空） | LLM 密钥 |
| `AME_LLM_MODEL` | `glm-4-flash` | 摘要用的模型名 |
| `AME_RECALL_THRESHOLD` | `0.9` | 距离门控（越大越松） |
| `AME_RECALL_POOL` | `15` | 阶段 A 宽网召回数 |
| `AME_ALPHA` | `0.5` | 重排中 strength 的权重（0..1） |
| `AME_MIN_STRENGTH` | `0.05` | 丢弃实时强度低于此值的记忆 |
| `AME_DEDUP_THRESHOLD` | `0.45` | 存储时距离 ≤ 此值则合并进已有记忆 |
| `AME_HYBRID_ENABLE` | `1` | vector + BM25 混合召回（0 = 纯向量） |
| `AME_BM25_POOL` | `15` | 阶段 A BM25 召回数（混合模式） |
| `AME_RERANKER_ENABLE` | `0` | 融合后用 cross-encoder 精确重排（可选开启） |
| `AME_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder 模型（多语言） |
| `AME_RERANKER_POOL` | `0` | 重排多少条候选（0 = 全部） |
| `AME_BACKUP_ENABLE` | `1` | 定期安全快照记忆库 |
| `AME_BACKUP_INTERVAL_HOURS` | `6` | 快照频率 |
| `AME_BACKUP_KEEP` | `5` | 只保留最新的 N 份备份 |
| `AME_SESSION_TIMEOUT_HOURS` | `2` | 会话超过此时长无心跳即判为过期 |

---

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```

测试跑在哈希兜底 embedder 上（不下载模型），所以快、CI 友好。覆盖：创建、精确匹配召回、门控、去重合并、艾宾浩斯衰减、召回时强化、向量维度一致性、混合 RRF 救回/归一化，以及重排兜底 + mock 提升行为。

---

## 许可证

MIT
