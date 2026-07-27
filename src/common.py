from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import requests


LOG = logging.getLogger("ai_talent_collection")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _stable_payload(
    method: str,
    url: str,
    params: Optional[Mapping[str, Any]],
    body: Any,
) -> bytes:
    safe_params = {
        str(key): value
        for key, value in (params or {}).items()
        if str(key).lower() not in {"api_key", "apikey", "key"}
    }
    value = {"method": method.upper(), "url": url, "params": safe_params, "body": body}
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


@dataclass
class CachedResponse:
    content: bytes
    retrieved_at: str
    cache_path: Path

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    def text(self) -> str:
        return self.content.decode("utf-8")


class RawCacheClient:
    """Rate-limited HTTP client that persists bytes before parsing."""

    def __init__(
        self,
        source: str,
        raw_root: str | Path,
        user_agent: str,
        delay_seconds: float,
        timeout_seconds: float = 90,
        max_retries: int = 6,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.source = source
        self.root = Path(raw_root) / source
        self.root.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = float(delay_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        if headers:
            self.session.headers.update(dict(headers))

    def _wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        extension: str = "json",
    ) -> CachedResponse:
        digest = hashlib.sha256(_stable_payload(method, url, params, json_body)).hexdigest()
        request_dir = self.root / digest[:2] / digest
        response_path = request_dir / f"response.{extension}"
        metadata_path = request_dir / "metadata.json"
        if response_path.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            return CachedResponse(response_path.read_bytes(), meta["retrieved_at"], response_path)

        request_dir.mkdir(parents=True, exist_ok=True)
        safe_params = {
            str(key): value
            for key, value in (params or {}).items()
            if str(key).lower() not in {"api_key", "apikey", "key"}
        }
        for attempt in range(self.max_retries + 1):
            self._wait()
            retrieved_at = utc_now()
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=self.timeout_seconds,
            )
            self.last_request_at = time.monotonic()

            # Preserve every network response before inspecting status or parsing.
            stamp = (
                retrieved_at.replace("-", "")
                .replace(":", "")
                .replace("+", "_")
                .replace(".", "")
            )
            attempt_stem = f"attempt-{attempt:02d}-{stamp}-{response.status_code}"
            attempt_path = request_dir / f"{attempt_stem}.{extension}"
            _atomic_write(attempt_path, response.content)
            attempt_meta = {
                "source": self.source,
                "method": method.upper(),
                "url": url,
                "params": safe_params,
                "json_body": json_body,
                "status_code": response.status_code,
                "retrieved_at": retrieved_at,
                "attempt": attempt,
            }
            _atomic_write(
                request_dir / f"{attempt_stem}.metadata.json",
                json.dumps(attempt_meta, indent=2, sort_keys=True).encode("utf-8"),
            )

            if 200 <= response.status_code < 300:
                _atomic_write(response_path, response.content)
                _atomic_write(
                    metadata_path,
                    json.dumps(attempt_meta, indent=2, sort_keys=True).encode("utf-8"),
                )
                return CachedResponse(response.content, retrieved_at, response_path)

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2**attempt)
                delay += random.random() * 0.25
                LOG.warning(
                    "source=%s status=%s retry=%s wait=%.2f",
                    self.source,
                    response.status_code,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise RuntimeError(f"{self.source} request failed after retries: {url}")


def _atomic_write(path: Path, content: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)
