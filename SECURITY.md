# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in agent-memory-engine, please report
it responsibly — **do not open a public issue**.

Instead, use GitHub's private vulnerability reporting (the **"Report a
vulnerability"** button on the repository's **Security** tab), or email the
maintainer at `ljftwq-dev@users.noreply.github.com`.

Please include:

- A description of the issue and its potential impact
- Steps to reproduce (a proof of concept if possible)
- The affected version

## Scope

The engine is designed to run **locally**: a single SQLite file and a
localhost-only HTTP server. The main risk surface is:

- Malformed input to the HTTP API (`/recall`, `/remember`, `/search`, …)
- The embedding / reranker model loading path
- The underlying `sqlite-vec` / SQLite stack

Anything that reaches a **remote service** you configure yourself (an
embedding or LLM endpoint via `AME_LLM_BASE_URL`, etc.) is outside this
project's scope — that's between you and your provider.

## Response

This is a personal project maintained in spare time. I'll acknowledge reports
within a few days and aim to ship a fix promptly for confirmed issues, with a
credit in the release notes (unless you prefer to stay anonymous).
