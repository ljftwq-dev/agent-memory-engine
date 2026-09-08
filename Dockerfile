# Agent Memory Engine — self-contained, local-first memory for coding agents.
#
# Build:  docker build -t agent-memory-engine .
# Run:    docker run -p 127.0.0.1:8765:8765 -v ame-data:/data agent-memory-engine
# (publish to localhost only; if you must expose further, set AME_API_TOKEN)
# The SQLite memory DB persists in the ame-data volume.
FROM python:3.11-slim

WORKDIR /app

# sqlite-vec ships prebuilt wheels, so no build tools are needed.
COPY pyproject.toml README.md ./
COPY engine ./engine
RUN pip install --no-cache-dir ".[all]"

# Persist the memory DB to a mounted volume; bind to all interfaces inside
# the container (publish only what you need with -p).
ENV AME_DB_PATH=/data/memory.db
VOLUME ["/data"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)" || sys.exit(1)

CMD ["python", "-m", "engine.server", "--host", "0.0.0.0", "--port", "8765"]
