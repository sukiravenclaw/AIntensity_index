"""Full per-person DBLP publication history for the training backtrack.

The venue-year seed (``dblp.collect_authorships``) only sees 2024-2025 papers.
To place a person at their PhD-era lab (e.g. Zhilin Yang at CMU ~2015-2019) we
need their *entire* publication timeline with co-authors. DBLP exposes that per
person at ``https://dblp.org/pid/{pid}.xml``.

Output is a long-format authorship frame (one row per work-author) covering the
union of every fetched person's works, deduplicated on ``(dblp_work_id,
author_dblp_pid)``. Works co-authored by several fetched persons collapse to one
set of rows, so the co-authorship graph is consistent regardless of fetch order.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.dblp_person")

HISTORY_COLUMNS = [
    "dblp_work_id",
    "author_dblp_pid",
    "author_name",
    "author_orcid",
    "author_ordinal",
    "author_position",
    "publication_year",
    "venue_raw",
    "retrieved_at",
    "source",
]

_PUB_TAGS = {"article", "inproceedings", "proceedings", "incollection", "book"}


def collect(cfg: Dict[str, Any], pids: Iterable[str]) -> pd.DataFrame:
    if not cfg.get("dblp_person", {}).get("enabled", True):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    # Preserve caller order (callers pass high-signal authors first) while
    # de-duplicating, so a max_pids cap keeps the most important authors.
    seen: set = set()
    unique_pids: List[str] = []
    for p in pids:
        if p and p not in seen:
            seen.add(p)
            unique_pids.append(p)
    max_pids = cfg.get("dblp_person", {}).get("max_pids")
    if max_pids:
        unique_pids = unique_pids[: int(max_pids)]

    endpoint = cfg.get("dblp_person", {}).get("endpoint", "https://dblp.org/pid").rstrip("/")
    client = _client(cfg)

    rows: List[Dict[str, Any]] = []
    fetched = 0
    failed = 0
    for pid in unique_pids:
        url = f"{endpoint}/{pid}.xml"
        try:
            raw = client.request("GET", url, extension="xml")
        except Exception as exc:  # 404 for merged/renamed PIDs, transient errors
            failed += 1
            LOG.warning("stage=dblp_person pid=%s fetch_failed=%s", pid, exc)
            continue
        try:
            parsed = _parse_person_xml(raw.content, raw.retrieved_at)
        except ET.ParseError as exc:
            failed += 1
            LOG.warning("stage=dblp_person pid=%s parse_failed=%s", pid, exc)
            continue
        rows.extend(parsed)
        fetched += 1

    frame = _finalize(rows)
    LOG.info(
        "stage=dblp_person pids=%d fetched=%d failed=%d works=%d rows=%d",
        len(unique_pids),
        fetched,
        failed,
        frame["dblp_work_id"].nunique() if not frame.empty else 0,
        len(frame),
    )
    return frame


def _parse_person_xml(content: bytes, retrieved_at: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(content)
    rows: List[Dict[str, Any]] = []
    for record in root.iter("r"):
        for pub in record:
            if pub.tag not in _PUB_TAGS:
                continue
            key = pub.get("key")
            if not key:
                continue
            work_id = f"https://dblp.org/rec/{key}"
            year = _text_int(pub.find("year"))
            venue = _first_text(pub, ("booktitle", "journal"))
            authors = [a for a in pub.findall("author")]
            for ordinal, author in enumerate(authors, start=1):
                pid = author.get("pid")
                if not pid:
                    continue
                rows.append(
                    {
                        "dblp_work_id": work_id,
                        "author_dblp_pid": pid.strip("/"),
                        "author_name": (author.text or "").strip() or None,
                        "author_orcid": author.get("orcid"),
                        "author_ordinal": ordinal,
                        "publication_year": year,
                        "venue_raw": venue,
                        "retrieved_at": retrieved_at,
                        "source": "dblp_person",
                    }
                )
    return rows


def _finalize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(["dblp_work_id", "author_dblp_pid"])
    max_ord = frame.groupby("dblp_work_id")["author_ordinal"].transform("max")
    position = pd.Series("middle", index=frame.index)
    position = position.mask(frame["author_ordinal"] == 1, "first")
    position = position.mask((frame["author_ordinal"] == max_ord) & (max_ord > 1), "last")
    frame["author_position"] = position
    return frame[HISTORY_COLUMNS].reset_index(drop=True)


def _first_text(element: ET.Element, tags: Iterable[str]) -> Optional[str]:
    for tag in tags:
        found = element.find(tag)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return None


def _text_int(element: Optional[ET.Element]) -> Optional[int]:
    if element is None or not (element.text or "").strip():
        return None
    try:
        return int(element.text.strip())
    except ValueError:
        return None


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "dblp_person",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate.get("dblp_person_seconds", rate.get("dblp_seconds", 2.0)),
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/xml"},
    )
