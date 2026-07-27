from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.openreview")


def collect(cfg: Dict[str, Any]) -> pd.DataFrame:
    columns = [
        "openreview_id",
        "forum",
        "title",
        "doi",
        "arxiv_id",
        "publication_year",
        "venue_normalized",
        "venue_raw",
        "decision",
        "review_scores",
        "review_score_count",
        "retrieved_at",
        "source",
    ]
    if not cfg["openreview"].get("enabled", True):
        return pd.DataFrame(columns=columns)

    client = _client(cfg)
    start_year = int(cfg["window"]["start"][:4])
    end_year = int(cfg["window"]["end"][:4])
    page_size = int(cfg["openreview"].get("page_size", 1000))
    rows: List[Dict[str, Any]] = []
    for venue, details in cfg["venues"].items():
        pattern = details.get("openreview_group")
        if not pattern:
            continue
        for year in range(start_year, end_year + 1):
            group = pattern.format(year=year)
            invitation = f"{group}/-/Submission"
            offset = 0
            while True:
                try:
                    raw = client.request(
                        "GET",
                        f"{cfg['openreview']['endpoint'].rstrip('/')}/notes",
                        params={
                            "invitation": invitation,
                            "details": "replies",
                            "limit": page_size,
                            "offset": offset,
                        },
                    )
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status in {400, 404}:
                        LOG.info(
                            "stage=openreview venue=%s year=%s unavailable_status=%s",
                            venue,
                            year,
                            status,
                        )
                        break
                    raise
                payload = raw.json()
                notes = payload.get("notes", [])
                for note in notes:
                    rows.append(_parse_submission(note, venue, year, raw.retrieved_at))
                LOG.info(
                    "stage=openreview venue=%s year=%s offset=%d rows=%d",
                    venue,
                    year,
                    offset,
                    len(notes),
                )
                if len(notes) < page_size:
                    break
                offset += len(notes)
    return pd.DataFrame(rows, columns=columns).drop_duplicates("openreview_id")


def _parse_submission(
    note: Dict[str, Any],
    venue: str,
    year: int,
    retrieved_at: str,
) -> Dict[str, Any]:
    content = note.get("content") or {}
    replies = (note.get("details") or {}).get("replies") or []
    decision_texts: List[str] = []
    scores: List[float] = []
    for reply in replies:
        reply_content = reply.get("content") or {}
        for key in ("decision", "venue", "recommendation"):
            value = _content_value(reply_content.get(key))
            if value:
                decision_texts.append(str(value))
        for key, value in reply_content.items():
            if "rating" in key.lower() or key.lower() in {"score", "recommendation"}:
                numeric = _first_number(_content_value(value))
                if numeric is not None:
                    scores.append(numeric)
    venue_raw = _content_value(content.get("venue")) or _content_value(content.get("venueid"))
    if venue_raw:
        decision_texts.append(str(venue_raw))
    identifiers = " ".join(
        str(_content_value(content.get(key)) or "")
        for key in ("doi", "arxiv", "arxiv_id", "pdf")
    )
    return {
        "openreview_id": note.get("id"),
        "forum": note.get("forum") or note.get("id"),
        "title": _content_value(content.get("title")),
        "doi": _normalize_doi(_content_value(content.get("doi"))),
        "arxiv_id": _extract_arxiv_id(identifiers),
        "publication_year": year,
        "venue_normalized": venue,
        "venue_raw": venue_raw or venue,
        "decision": classify_decision(" | ".join(decision_texts)),
        "review_scores": scores,
        "review_score_count": len(scores),
        "retrieved_at": retrieved_at,
        "source": "openreview",
    }


def classify_decision(text: str) -> str:
    normalized = text.casefold()
    if "reject" in normalized or "withdraw" in normalized:
        return "rejected"
    if "oral" in normalized:
        return "oral"
    if "spotlight" in normalized:
        return "spotlight"
    if "poster" in normalized:
        return "poster"
    if "accept" in normalized:
        return "poster"
    return "unknown"


def _content_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _first_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _extract_arxiv_id(text: str) -> Optional[str]:
    match = re.search(r"(?:arxiv[:./\s]+)?(\d{4}\.\d{4,5})(?:v\d+)?", text, re.I)
    return match.group(1) if match else None


def _normalize_doi(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", text, flags=re.I)
    return text.lower() or None


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "openreview",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate["openreview_seconds"],
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/json"},
    )
