"""Bearer-token auth tests (issue #4): AME_API_TOKEN gate on the HTTP API."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from engine import server


def _request(url, headers=None, method="GET", data=None):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    if data is not None:
        req.data = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


@pytest.fixture()
def live_server(monkeypatch):
    """Real HTTP server on an ephemeral port, with a token configured."""
    monkeypatch.setenv("AME_API_TOKEN", "secret-token-123")
    # Warm the embedder (first remember() lazily imports torch etc., which can
    # exceed the HTTP timeout on slow machines — not what this test measures).
    from engine.remember import remember as _warm
    _warm(topic="warmup", summary="embedder warmup for auth tests")
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


class TestUnitGate:
    def test_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("AME_API_TOKEN", raising=False)
        assert server._bearer_ok("/recall", None) is True
        assert server._bearer_ok("/remember", None) is True

    def test_exempt_paths_stay_open(self, monkeypatch):
        monkeypatch.setenv("AME_API_TOKEN", "tok")
        assert server._bearer_ok("/health", None) is True
        assert server._bearer_ok("/metrics", None) is True

    def test_protected_paths(self, monkeypatch):
        monkeypatch.setenv("AME_API_TOKEN", "tok")
        assert server._bearer_ok("/recall", None) is False
        assert server._bearer_ok("/recall", "Bearer wrong") is False
        assert server._bearer_ok("/recall", "Basic dG9r") is False
        assert server._bearer_ok("/recall", "Bearer tok") is True
        assert server._bearer_ok("/remember", "bearer tok") is True  # scheme is case-insensitive


class TestLiveServer:
    def test_missing_or_wrong_token_401(self, live_server):
        code, body = _request(live_server + "/recent?k=1")
        assert code == 401 and body["ok"] is False
        code, body = _request(live_server + "/recall?q=test",
                              headers={"Authorization": "Bearer nope"})
        assert code == 401 and body["ok"] is False

    def test_valid_token_passes(self, live_server):
        hdr = {"Authorization": "Bearer secret-token-123"}
        code, body = _request(live_server + "/recent?k=1", headers=hdr)
        assert code == 200 and body["ok"] is True
        code, body = _request(live_server + "/remember", method="POST",
                              headers=hdr,
                              data={"topic": "auth", "summary": "token gate works"})
        assert code == 200 and body["ok"] is True

    def test_exempt_endpoints_open_without_token(self, live_server):
        code, body = _request(live_server + "/health")
        assert code == 200 and body["ok"] is True
        code, body = _request(live_server + "/metrics")
        assert code == 200 and body["ok"] is True
