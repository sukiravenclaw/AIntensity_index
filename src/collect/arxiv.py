from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.arxiv")
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def collect(cfg: Dict[str, Any]) -> pd.DataFrame:
    columns = [
        "arxiv_id",
        "title",
        "doi",
        "publication_date",
        "publication_year",
        "updated_at",
        "categories",
        "primary_category",
        "retrieved_at",
        "source",
    ]
    if not cfg["arxiv"].get("enabled", True):
        return pd.DataFrame(columns=columns)

    categories = cfg["arxiv"]["categories"]
    window_start = date.fromisoformat(cfg["window"]["start"])
    window_end = date.fromisoformat(cfg["window"]["end"])
    page_size = int(cfg["arxiv"].get("page_size", 1000))
    max_pages = cfg["arxiv"].get("max_pages")
    client = _client(cfg)
    rows: List[Dict[str, Any]] = []
    page = 0
    stop = False
    # Monthly, per-category slices avoid the API's practical large-query result
    # ceiling while retaining complete coverage. Cross-listed papers are
    # deduplicated by arXiv ID after collection.
    for category in categories:
        if stop:
            break
        for slice_start, slice_end in _month_slices(window_start, window_end):
            start_index = 0
            total = None
            search_query = (
                f"cat:{category} AND submittedDate:"
                f"[{slice_start.strftime('%Y%m%d')}0000 TO "
                f"{slice_end.strftime('%Y%m%d')}2359]"
            )
            while total is None or start_index < total:
                if max_pages is not None and page >= int(max_pages):
                    LOG.warning(
                        "stage=arxiv stopped_at_max_pages pages=%d total=%s",
                        page,
                        total,
                    )
                    stop = True
                    break
                raw = client.request(
                    "GET",
                    "https://export.arxiv.org/api/query",
                    params={
                        "search_query": search_query,
                        "start": start_index,
                        "max_results": page_size,
                        "sortBy": "submittedDate",
                        "sortOrder": "ascending",
                    },
                    extension="xml",
                )
                root = ET.fromstring(raw.content)
                if total is None:
                    total_text = root.findtext(f"{OPENSEARCH}totalResults")
                    total = int(total_text or 0)
                entries = root.findall(f"{ATOM}entry")
                for entry in entries:
                    parsed = _parse_entry(entry, raw.retrieved_at, set(categories))
                    if parsed:
                        rows.append(parsed)
                LOG.info(
                    "stage=arxiv category=%s month=%s page=%d start=%d rows=%d total=%s",
                    category,
                    slice_start.strftime("%Y-%m"),
                    page,
                    start_index,
                    len(entries),
                    total,
                )
                if not entries:
                    break
                start_index += len(entries)
                page += 1
            if stop:
                break
    return pd.DataFrame(rows, columns=columns).drop_duplicates("arxiv_id")


def _parse_entry(
    entry: ET.Element,
    retrieved_at: str,
    target_categories: set,
) -> Optional[Dict[str, Any]]:
    identifier = entry.findtext(f"{ATOM}id") or ""
    arxiv_id = _extract_arxiv_id(identifier)
    categories = [
        element.attrib.get("term")
        for element in entry.findall(f"{ATOM}category")
        if element.attrib.get("term")
    ]
    if not set(categories).intersection(target_categories):
        return None
    primary = entry.find(f"{ARXIV}primary_category")
    published = entry.findtext(f"{ATOM}published")
    return {
        "arxiv_id": arxiv_id,
        "title": _collapse(entry.findtext(f"{ATOM}title")),
        "doi": _normalize_doi(entry.findtext(f"{ARXIV}doi")),
        "publication_date": published[:10] if published else None,
        "publication_year": int(published[:4]) if published else None,
        "updated_at": entry.findtext(f"{ATOM}updated"),
        "categories": categories,
        "primary_category": primary.attrib.get("term") if primary is not None else None,
        "retrieved_at": retrieved_at,
        "source": "arxiv",
    }


def _collapse(value: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", value).strip() if value else None


def _extract_arxiv_id(value: str) -> Optional[str]:
    match = re.search(r"(?:abs/)?(\d{4}\.\d{4,5})(?:v\d+)?$", value)
    return match.group(1) if match else None


def _normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", value.strip(), flags=re.I).lower()


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "arxiv",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate["arxiv_seconds"],
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/atom+xml"},
    )


def _month_slices(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        last_day = monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        yield max(start, current), min(end, month_end)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
