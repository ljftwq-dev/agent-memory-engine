"""dataset.py - evaluation set for the retrieval benchmark.

40 coding-agent-style memories across 5 domains (Python, Git/Tools, ML/DL,
Quant/Finance, Math) + 18 queries with hand-labeled **graded relevance**
(2 = highly relevant, 1 = partially relevant, absent = irrelevant).

Design goals:
- Realistic: mirrors what a coding agent (opencode / claude-code) would store.
- Distractor-rich: most memories are NOT relevant to any given query, so a
  method that just "returns semantically adjacent" junk scores poorly.
- Hard cases: some queries are keyword-ambiguous (pure vector misses them) but
  a cross-encoder nails them -> that's where reranking should show gains.

Memories are seeded in list order with dedup disabled, so a memory's rowid ==
its 1-based index here. Queries reference relevance by that index.

This is an IR benchmark, NOT a secret/private corpus: all content is generic
textbook material, safe to ship in the repo.
"""

# (topic, summary) - the two fields the engine embeds & stores.
MEMORIES = [
    # --- Python (1-8) ---
    ("pytest fixtures",
     "Discussed pytest fixtures: function scope recreates the fixture per test, "
     "session scope shares one instance across the whole run. Put shared fixtures "
     "in conftest.py. Fixtures keep tests isolated and DRY."),
    ("Python decorators",
     "A decorator wraps a function to add behavior without changing its code: "
     "@decorator above def f(). The decorator is a callable returning a callable. "
     "Use functools.wraps to preserve metadata."),
    ("Python async/await",
     "async def declares a coroutine; await suspends it until the awaited "
     "future resolves, yielding control to the event loop. Great for IO-bound "
     "concurrency (network, disk) without threads."),
    ("Python generators",
     "A generator yields values lazily one at a time with 'yield', pausing state "
     "between calls. Memory-efficient for large streams. Use itertools.chain to "
     "compose them."),
    ("Python context managers",
     "'with' statements guarantee cleanup via __enter__/__exit__. Use "
     "contextlib.contextmanager on a generator for a lightweight version. Always "
     "use 'with open(...)' for files."),
    ("list comprehensions",
     "[expr for x in iter if cond] builds a list concisely. Prefer over manual "
     "loops. Nested comprehensions get unreadable fast - stop at two levels."),
    ("Python GIL",
     "The Global Interpreter Lock lets only one thread run Python bytecode at a "
     "time, so threading doesn't speed up CPU-bound work. Use multiprocessing or "
     "C extensions to bypass it."),
    ("virtual environments",
     "venv creates an isolated Python environment per project so dependencies "
     "don't clash. 'python -m venv .venv' then activate. Pin versions in "
     "requirements.txt for reproducibility."),

    # --- Git / Tools (9-14) ---
    ("Git interactive rebase",
     "git rebase -i HEAD~3 opens an editor to squash, reword, or reorder the "
     "last 3 commits. Never rebase commits already pushed to a shared branch. "
     "Use git reflog to recover from mistakes."),
    ("Git merge conflicts",
     "When two branches change the same lines, git pauses the merge and marks "
     "<<<<<<< ======= >>>>>>> blocks. Edit to resolve, git add, then commit. "
     "Pull regularly to keep conflicts small."),
    ("Git stash",
     "git stash temporarily shelves uncommitted changes so you can switch "
     "branches with a clean tree. git stash pop restores them. Useful before a "
     "pull or context switch."),
    ("Docker multi-stage builds",
     "Multi-stage Dockerfiles compile in one image then COPY only the artifacts "
     "into a slim final image, slashing the shipped size. Each FROM starts a "
     "stage; name them with AS."),
    ("SQLite basics",
     "SQLite is a serverless single-file relational DB. ACID, zero-config, great "
     "for prototyping and embedded use. Limited write concurrency - one writer "
     "at a time via a lock."),
    ("SQLite vector search (sqlite-vec)",
     "sqlite-vec adds a vec0 virtual table storing FLOAT[dim] embeddings; KNN via "
     "'WHERE embedding MATCH ? AND k = N'. Zero extra service - the index lives "
     "inside the .db file."),

    # --- ML / Deep Learning (15-23) ---
    ("Transformer self-attention",
     "Self-attention computes Q, K, V from inputs; attention weights = "
     "softmax(QK^T / sqrt(d_k)); output = weights * V. Every position attends to "
     "all others, avoiding RNN's long-path problem."),
    ("RNN backpropagation through time",
     "Training an RNN unrolls it across timesteps and backprops through each, "
     "causing vanishing/exploding gradients on long sequences. Gradient clipping "
     "and LSTM/GRU gates mitigate this."),
    ("Dropout regularization",
     "Dropout randomly zeroes a fraction of activations during training, forcing "
     "the network not to rely on any single neuron. Reduces overfitting. Disabled "
     "at inference."),
    ("Batch normalization",
     "BatchNorm normalizes layer activations per mini-batch to zero mean / unit "
     "variance, then scales and shifts. Stabilizes and speeds training, acts as "
     "mild regularization."),
    ("Gradient descent variants",
     "SGD with momentum, RMSProp, Adam. Adam combines momentum and per-parameter "
     "adaptive learning rates; usually the default choice. Tune the learning "
     "rate - it matters most."),
    ("Cross-entropy loss",
     "For classification, cross-entropy -sum(y log p) penalizes confident wrong "
     "predictions heavily. Paired with softmax outputs. Lower = better calibrated "
     "probabilities."),
    ("CNN convolution",
     "A conv layer slides small learnable kernels over the input, capturing local "
     "spatial patterns (edges, textures) with weight sharing and translation "
     "equivariance. Backbone of image models."),
    ("Q-learning",
     "Q-learning learns the action-value Q(s,a) by Bellman updates; the greedy "
     "policy picks argmax_a Q(s,a). Off-policy, model-free. epsilon-greedy "
     "explores while learning."),
    ("Policy gradient methods",
     "Policy gradients optimize the policy directly via REINFORCE: "
     "grad J = E[grad log pi(a|s) * return]. High variance - baselines and "
     "Actor-Critic (PPO/A2C) reduce it."),

    # --- Web / Systems (24-28) ---
    ("REST API design",
     "REST maps resources to URLs and uses HTTP verbs (GET/POST/PUT/DELETE) with "
     "status codes. Stateless, cacheable. Keep URLs noun-based and versioned "
     "(/v1/)."),
    ("SQL JOINs",
     "JOINs combine rows across tables on a key: INNER keeps matches only, LEFT "
     "keeps all left rows, FULL keeps all. Index the join columns for speed."),
    ("Database indexes",
     "An index is a sorted structure (usually B-tree) speeding lookups and "
     "range scans at the cost of slower writes and extra storage. Index columns "
     "used in WHERE/JOIN, not everything."),
    ("Caching strategies",
     "Cache repeated expensive results: in-memory (LRU), CDN, or Redis. Watch "
     "staleness and invalidation - cache bugs are subtle. Cache-aside is the "
     "common pattern."),
    ("Recursion and memoization",
     "Recursion solves a problem in terms of smaller subproblems; memoization "
     "caches subproblem results to avoid exponential recomputation. Classic on "
     "fibonacci, edit distance."),

    # --- Quant / Finance (29-35) ---
    ("Momentum factor",
     "The momentum factor bets that assets with recent positive returns keep "
     "outperforming in the near term (Jegadeesh-Titman). Long winners, short "
     "losers over a 3-12 month lookback."),
    ("Mean reversion",
     "Mean reversion assumes prices drift back toward a historical average; "
     "pairs trading and Ornstein-Uhlenbeck models exploit this. Opposite of "
     "momentum at short horizons."),
    ("Sharpe ratio",
     "Sharpe = (return - risk_free) / volatility measures risk-adjusted return. "
     "Higher is better. Annualize by multiplying by sqrt(252) for daily returns."),
    ("Black-Scholes option pricing",
     "Black-Scholes prices a European option from spot, strike, time, volatility, "
     "and risk-free rate. Assumes lognormal returns and constant vol. The Greeks "
     "are its partial derivatives."),
    ("Portfolio optimization",
     "Markowitz mean-variance optimization finds weights minimizing variance for "
     "a target return. Sensitive to estimated covariances - shrinkage helps. The "
     "efficient frontier is its output."),
    ("ARIMA time series",
     "ARIMA(p,d,q) models a series as auto-regressive + moving-average after "
     "d differencing steps to stationarity. Box-Jenkins: identify p,d,q via ACF/"
     "PACF, fit, check residuals."),
    ("Value at Risk (VaR)",
     "VaR is the loss not exceeded with confidence alpha (e.g. 95%) over a "
     "horizon. Historical, parametric, or Monte-Carlo. Criticized for ignoring "
     "tail shape - CVaR goes further."),

    # --- Math (36-40) ---
    ("Softmax gradient",
     "The Jacobian of softmax is diag(s) - s s^T. Combined with cross-entropy "
     "the gradient simplifies to (p - y), which is why they're paired in "
     "classification output layers."),
    ("SVD matrix decomposition",
     "SVD factors any matrix A = U S V^T. The singular values in S reveal rank "
     "and energy concentration; truncated SVD gives the best low-rank "
     "approximation (Eckart-Young)."),
    ("Probability distributions",
     "Key distributions: Normal (continuous, symmetric), Exponential (memoryless "
     "wait times), Poisson (event counts), Binomial (success counts). Match the "
     "distribution to the data-generating process."),
    ("Markov chains",
     "A Markov chain's next state depends only on the current state (memoryless). "
     "Characterized by a transition matrix; a stationary distribution pi satisfies "
     "pi = pi P."),
    ("Stochastic calculus (Ito's lemma)",
     "Ito's lemma is the chain rule for stochastic processes: it adds a "
     "+0.5 g'^2 sigma^2 dt term. Foundation of Black-Scholes and continuous-time "
     "finance."),
]


# Each query: (query_text, {memory_index: grade}) - grades 2 (high) / 1 (partial).
# memory_index is 1-based (matches the seeded rowid).
QUERIES = [
    ("how to share setup code across multiple tests",
     {1: 2}),
    ("squash and reorder my last few commits",
     {9: 2}),
    ("how are attention weights computed in transformers",
     {15: 2}),
    ("shelve uncommitted changes before switching branches",
     {11: 2}),
    ("run network requests without blocking the thread",
     {3: 2, 4: 1}),
    ("measure risk-adjusted return of a strategy",
     {31: 2}),
    ("resolve conflicting edits when merging two branches",
     {10: 2}),
    ("reduce overfitting in a neural network",
     {17: 2, 18: 1}),
    ("store embeddings and run nearest-neighbor search inside sqlite",
     {14: 2, 13: 1}),
    ("learn an optimal action policy from rewards",
     {22: 2, 23: 1}),
    ("price a european call option",
     {32: 2, 40: 1}),
    ("wrap a function to add behavior without editing it",
     {2: 2}),
    ("guarantee a file or resource is cleaned up after use",
     {5: 2}),
    ("strategy that bets recent winners keep winning",
     {29: 2}),
    ("produce values lazily one at a time from a stream",
     {4: 2}),
    ("compress a matrix to its most important components",
     {37: 2}),
    ("build a small docker image by copying only the build output",
     {12: 2}),
    ("speed up a recursive function that recomputes the same subproblems",
     {25: 2}),

    # --- Harder / colloquial queries (realistic user phrasings with low lexical
    # overlap to the target memory, or lexical traps). This is where a strong
    # bi-encoder starts to slip and a cross-encoder should pull ahead. ---
    ("stop my docker image from being bloated",
     {12: 2}),
    ("the effect where stocks that went up keep going up",
     {29: 2}),
    ("don't repeat the same setup boilerplate in every test file",
     {1: 2}),
    ("simplify a huge matrix keeping only what really matters",
     {37: 2}),
    ("why can't two python threads both run code at the same time",
     {7: 2}),
    ("how do transformers decide which input words to focus on",
     {15: 2}),
]
