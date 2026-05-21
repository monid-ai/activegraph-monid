"""Pure httpx wrapper around the monid HTTP API.

No activegraph imports here: this is a plain HTTP client with a small
on-disk response cache. The cache makes Phase 2 forks cheap and keeps
development loops fast.

Cache strategy:
    key  = sha256(method, path, body)
    file = ./fixtures/monid/{method}_{key}.json
    write on terminal success only (run COMPLETED with provider 2xx)

API surface used:
    POST /v1/discover     ->  {results: [...], query, count}
    POST /v1/inspect      ->  {provider, endpoint, description, input, ...}
    POST /v1/run          ->  200 sync result, or 202 -> poll /v1/runs/:id
    GET  /v1/runs/:runId  ->  {status, output, cost, providerResponse, ...}

Docs: https://docs.monid.ai/api/overview.html
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

import httpx


class MonidError(RuntimeError):
    """Raised when the monid API returns a non-success status."""


class MonidClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.monid.ai",
        cache_dir: str | pathlib.Path = "./fixtures/monid",
        request_timeout: float = 30.0,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=request_timeout,
        )
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API -------------------------------------------------------

    def discover(self, query: str, limit: int = 10) -> dict[str, Any]:
        return self._cached_post(
            "/v1/discover", {"query": query, "limit": limit}
        )

    def inspect(self, provider: str, endpoint: str) -> dict[str, Any]:
        return self._cached_post(
            "/v1/inspect", {"provider": provider, "endpoint": endpoint}
        )

    def run(
        self,
        provider: str,
        endpoint: str,
        input: dict[str, Any],
        poll_timeout: int = 120,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        """Execute a monid endpoint and return a normalized result.

        Handles both sync providers (200 with output inline) and async
        providers (202 with runId requiring polling). On success, returns:

            {
              "run_id":      ULID,
              "provider":    str,
              "endpoint":    str,
              "output":      provider data (list or dict, possibly None),
              "cost_usd":    float,
              "http_status": int,            # provider's HTTP status
              "status":      "COMPLETED",
              "cached":      bool,
            }
        """
        body = {"provider": provider, "endpoint": endpoint, "input": input}
        cache_key = self._key("run", body)
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

        resp = self._http.post("/v1/run", json=body)
        data = resp.json()

        if resp.status_code == 200:
            # Sync provider: results are already here.
            result = self._normalize(data)
        elif resp.status_code == 202:
            # Async provider: poll until terminal.
            run_id = data["runId"]
            result = self._poll_run(run_id, poll_timeout, poll_interval)
        else:
            raise MonidError(
                f"POST /v1/run failed: {resp.status_code} {data!r}"
            )

        # Only cache fully-successful terminal results.
        if (
            result["status"] == "COMPLETED"
            and result["http_status"] < 400
            and result["output"] is not None
        ):
            self._write_cache(cache_key, result)
        result["cached"] = False
        return result

    # ---- internals --------------------------------------------------------

    def _poll_run(
        self, run_id: str, poll_timeout: int, poll_interval: float
    ) -> dict[str, Any]:
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            resp = self._http.get(f"/v1/runs/{run_id}")
            if resp.status_code != 200:
                raise MonidError(
                    f"GET /v1/runs/{run_id} failed: {resp.status_code}"
                )
            data = resp.json()
            status = data.get("status")
            if status in ("COMPLETED", "FAILED"):
                return self._normalize(data)
            time.sleep(poll_interval)
        raise MonidError(f"monid run {run_id} did not complete in {poll_timeout}s")

    @staticmethod
    def _normalize(data: dict[str, Any]) -> dict[str, Any]:
        cost = data.get("cost") or data.get("billing", {}).get("reportedCost", {})
        # `cost.value` is sometimes micro-dollars and sometimes USD; the API
        # docs show both shapes. Prefer the run-level `cost.value` which is
        # documented as USD in /v1/runs/:id. We treat any value > 100 as
        # micro-dollars and convert.
        cost_value = float(cost.get("value", 0.0)) if cost else 0.0
        if cost.get("unit") == "MICRO_DOLLAR":
            cost_value /= 1_000_000
        provider_resp = data.get("providerResponse") or {}
        return {
            "run_id": data.get("runId"),
            "provider": data.get("provider"),
            "endpoint": data.get("endpoint"),
            "output": data.get("output"),
            "cost_usd": cost_value,
            "http_status": provider_resp.get("httpStatus", 200),
            "status": data.get("status", "COMPLETED"),
        }

    def _cached_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        key = self._key(path, body)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        resp = self._http.post(path, json=body)
        if resp.status_code >= 400:
            raise MonidError(
                f"POST {path} failed: {resp.status_code} {resp.text[:300]}"
            )
        data = resp.json()
        self._write_cache(key, data)
        return data

    def _key(self, path_or_method: str, body: dict[str, Any]) -> str:
        material = json.dumps(
            [path_or_method, body], sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:24]

    def _cache_path(self, key: str) -> pathlib.Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        self._cache_path(key).write_text(json.dumps(value, indent=2, default=str))

    def close(self) -> None:
        self._http.close()
