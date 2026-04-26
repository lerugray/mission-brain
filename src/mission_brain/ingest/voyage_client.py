"""Voyage AI embeddings client — cloud egress.

Wrapper around Voyage's ``/v1/embeddings`` endpoint. Requests
pass through :func:`mission_brain.ingest.network_guard.assert_local`
with ``api.voyageai.com`` on the allowlist — the guard still
fires, cloud egress is explicit and narrow.

Non-generative: returns vectors only. Importable from
``src/mission_brain/query/`` without violating the §2.4 generation
boundary.
"""

from __future__ import annotations

import os
import time

import httpx

from mission_brain.ingest.network_guard import assert_local

__all__ = ["DEFAULT_EMBED_MODEL", "VOYAGE_HOST", "VoyageEmbedder"]

VOYAGE_HOST = "api.voyageai.com"
DEFAULT_BASE_URL = f"https://{VOYAGE_HOST}"
DEFAULT_EMBED_MODEL = "voyage-3-lite"
BATCH_SIZE = 128
# Voyage free-tier limits requests/min; build_index over a 480-page
# vault sends 4 batches in rapid succession and the trailing ones
# 429. Retry with exponential backoff + Retry-After header so the
# first-time index-build completes without manual intervention.
MAX_RETRIES = 5
RETRY_BACKOFF_BASE_SEC = 2.0
RETRY_MAX_WAIT_SEC = 60.0


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "VOYAGE_API_KEY not set — put it in .env or export it."
            )
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed each text in *texts* and return their vectors."""
        url = f"{self.base_url}/v1/embeddings"
        chosen_model = model or os.environ.get(
            "MISSION_BRAIN_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            assert_local(url, allowlist=[VOYAGE_HOST])
            payload = self._post_with_retry(
                url=url,
                headers=headers,
                json_body={"input": batch, "model": chosen_model},
            )
            for item in sorted(payload["data"], key=lambda x: x["index"]):
                vectors.append(list(item["embedding"]))
        return vectors

    def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        json_body: dict,
    ) -> dict:
        """POST with retry-on-429 + exponential backoff.

        Honors Retry-After header when present (Voyage typically sends
        seconds). Falls back to exponential backoff (base * 2^attempt)
        capped at RETRY_MAX_WAIT_SEC when the header is missing.
        Re-raises HTTPStatusError after MAX_RETRIES attempts so the
        caller still sees the failure rather than silently looping.
        """
        last_response: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            response = httpx.post(
                url,
                json=json_body,
                headers=headers,
                timeout=self.timeout,
            )
            last_response = response
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            wait_seconds = RETRY_BACKOFF_BASE_SEC * (2**attempt)
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header is not None:
                try:
                    wait_seconds = max(wait_seconds, float(retry_after_header))
                except ValueError:
                    pass
            wait_seconds = min(wait_seconds, RETRY_MAX_WAIT_SEC)
            time.sleep(wait_seconds)

        assert last_response is not None
        last_response.raise_for_status()
        return last_response.json()
