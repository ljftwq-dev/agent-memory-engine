# One-line installer for Agent Memory Engine (Windows).
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/ljftwq-dev/agent-memory-engine/main/install.ps1 | iex
#
# Installs the PyPI package with real embeddings (BGE-m3) + reranker + MCP.
# Override Python with $env:PYTHON=... ; override extras with $env:EXTRAS=all.
$ErrorActionPreference = "Stop"

Write-Host "=== Agent Memory Engine installer ==="
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$extras = if ($env:EXTRAS) { $env:EXTRAS } else { "all" }

if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: '$py' not found. Install Python 3.10+ from python.org or set `$env:PYTHON='C:\path\to\python.exe'."
    exit 1
}

Write-Host "Using: $(& $py --version)"
& $py -m pip install --upgrade "agent-memory-engine-ljf[$extras]"

Write-Host @"

Installed. Next steps:

  # 1. Start the HTTP server (loads BGE-m3 once, stays up on :8765)
  agent-memory

  # 2. (Optional) point an MCP-capable IDE at the engine:
  agent-memory-mcp      # needs the HTTP server running first

  # 3. Verify it's up (curl.exe, not the Invoke-WebRequest alias):
  curl.exe -s http://127.0.0.1:8765/health

Docs: https://github.com/ljftwq-dev/agent-memory-engine
"@
