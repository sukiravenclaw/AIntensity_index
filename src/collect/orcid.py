"""Current-employer signal from the free ORCID public API.

``GET https://pub.orcid.org/v3.0/{orcid}/employments`` returns structured, dated
employment summaries. Unlike the S2 affiliation blob this is dated, so we can
identify the *most recent* employer — the best free current-employer signal.
ORCIDs come from CSRankings and (later) Wikidata. No API key is required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.orcid")

EMPLOYMENT_COLUMNS = [
    "orcid",
    "employer_name",
    "role",
    "start_year",
    "end_year",
    "is_current",
    "retrieved_at",
    "source",
]


def collect(cfg: Dict[str, Any], orcids: Iterable[str]) -> pd.DataFrame:
    if not cfg.get("orcid", {}).get("enabled", True):
        return pd.DataFrame(columns=EMPLOYMENT_COLUMNS)

    unique = sorted({_clean_orcid(o) for o in orcids if o})
    unique = [o for o in unique if o]
    if not unique:
        return pd.DataFrame(columns=EMPLOYMENT_COLUMNS)

    endpoint = cfg.get("orcid", {}).get("endpoint", "https://pub.orcid.org/v3.0").rstrip("/")
    client = _client(cfg)
    rows: List[Dict[str, Any]] = []
    failed = 0
    for orcid in unique:
        try:
            raw = client.request("GET", f"{endpoint}/{orcid}/employments")
        except Exception as exc:
            failed += 1
            LOG.warning("stage=orcid orcid=%s fetch_failed=%s", orcid, exc)
            continue
        rows.extend(_parse_employments(orcid, raw.json(), raw.retrieved_at))

    frame = pd.DataFrame(rows, columns=EMPLOYMENT_COLUMNS)
    LOG.info("stage=orcid orcids=%d failed=%d rows=%d", len(unique), failed, len(frame))
    return frame


def _parse_employments(orcid: str, payload: Any, retrieved_at: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = (payload or {}).get("affiliation-group") or []
    for group in groups:
        for summary in group.get("summaries") or []:
            emp = summary.get("employment-summary") or {}
            org = (emp.get("organization") or {}).get("name")
            if not org:
                continue
            start_year = _year(emp.get("start-date"))
            end_date = emp.get("end-date")
            end_year = _year(end_date)
            rows.append(
                {
                    "orcid": orcid,
                    "employer_name": org,
                    "role": emp.get("role-title"),
                    "start_year": start_year,
                    "end_year": end_year,
                    "is_current": end_date is None,
                    "retrieved_at": retrieved_at,
                    "source": "orcid",
                }
            )
    return rows


def most_recent_employer(employments: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one most-recent employer per ORCID.

    Preference: a current (open-ended) role, else the latest start_year.
    """
    if employments.empty:
        return pd.DataFrame(columns=["orcid", "employer_name", "start_year", "is_current"])
    df = employments.copy()
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").fillna(-1)
    df = df.sort_values(["is_current", "start_year"], ascending=[False, False])
    top = df.groupby("orcid", as_index=False).first()
    return top[["orcid", "employer_name", "start_year", "is_current"]]


def _year(date_obj: Optional[Dict[str, Any]]) -> Optional[int]:
    if not date_obj:
        return None
    year = (date_obj.get("year") or {}).get("value")
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _clean_orcid(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    # Accept bare id or a full https://orcid.org/xxxx-xxxx-xxxx-xxxx URL.
    if "orcid.org/" in text:
        text = text.split("orcid.org/", 1)[-1]
    text = text.strip("/").upper()
    return text if is_valid_orcid(text) else None


def is_valid_orcid(orcid: str) -> bool:
    """Format + ISO 7064 MOD 11-2 checksum check (filters placeholder/garbage IDs)."""
    if not orcid or len(orcid) != 19:
        return False
    digits = orcid.replace("-", "")
    if len(digits) != 16 or not digits[:15].isdigit():
        return False
    total = 0
    for ch in digits[:15]:
        total = (total + int(ch)) * 2
    check = (12 - (total % 11)) % 11
    expected = "X" if check == 10 else str(check)
    return digits[15] == expected


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "orcid",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate.get("orcid_seconds", 0.5),
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/json"},
    )
