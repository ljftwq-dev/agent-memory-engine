#!/usr/bin/env bash
# One-line installer for Agent Memory Engine.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ljftwq-dev/agent-memory-engine/main/install.sh | bash
#
# Installs the PyPI package with real embeddings (BGE-m3) + reranker + MCP.
# Override Python with PYTHON=python3.11 ... ; override extras with EXTRAS=all.
set -e

echo "=== Agent Memory Engine installer ==="
PY="${PYTHON:-python3}"
EXTRAS="${EXTRAS:-all}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: '$PY' not found. Install Python 3.9+ or set PYTHON=/path/to/python."
    exit 1
fi

echo "Using: $($PY --version)"
$PY -m pip install --upgrade "agent-memory-engine-ljf[$EXTRAS]"

cat <<EOF

Installed. Next steps:

  # 1. Start the HTTP server (loads BGE-m3 once, stays up on :8765)
  agent-memory

  # 2. (Optional) point an MCP-capable IDE at the engine:
  agent-memory-mcp      # needs the HTTP server running first

  # 3. Verify it's up:
  curl -s http://127.0.0.1:8765/health

Docs: https://github.com/ljftwq-dev/agent-memory-engine
EOF
