from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from src.common import RawCacheClient
from src.normalize.text import normalize_text


LOG = logging.getLogger("ai_talent_collection.semantic_scholar")


def enrich(
    cfg: Dict[str, Any],
    works: pd.DataFrame,
    authorships: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add S2 paper/author IDs and fill only missing citation counts."""
    interim_columns = [
        "work_id",
        "s2_paper_id",
        "citation_count",
        "external_ids",
        "retrieved_at",
        "source",
    ]
    if works.empty or not cfg["semantic_scholar"].get("enabled", True):
        return works, authorships, pd.DataFrame(columns=interim_columns)
    if not cfg.get("_s2_api_key"):
        LOG.warning(
            "stage=semantic_scholar skipped=true reason=missing_api_key "
            "env=%s",
            cfg["semantic_scholar"].get("api_key_env", "SEMANTIC_SCHOLAR_API_KEY"),
        )
        return works, authorships, pd.DataFrame(columns=interim_columns)

    lookup: Dict[str, str] = {}
    for row in works.itertuples():
        if row.doi:
            lookup[f"DOI:{row.doi}"] = row.work_id
        elif row.arxiv_id:
            lookup[f"ARXIV:{row.arxiv_id}"] = row.work_id
    client = _client(cfg)
    results: Dict[str, Tuple[Dict[str, Any], str]] = {}
    batch_size = min(500, int(cfg["semantic_scholar"].get("batch_size", 500)))
    identifiers = list(lookup)
    for chunk in _chunks(identifiers, batch_size):
        raw = client.request(
            "POST",
            f"{cfg['semantic_scholar']['endpoint'].rstrip('/')}/paper/batch",
            params={
                "fields": "title,externalIds,citationCount,authors",
            },
            json_body={"ids": chunk},
        )
        payload = raw.json()
        for requested, paper in zip(chunk, payload):
            if paper:
                results[lookup[requested]] = (paper, raw.retrieved_at)

    works = works.copy()
    authorships = authorships.copy()
    interim_rows: List[Dict[str, Any]] = []
    for work_id, (paper, retrieved_at) in results.items():
        mask = works["work_id"] == work_id
        works.loc[mask, "s2_paper_id"] = paper.get("paperId")
        if works.loc[mask, "cited_by_count"].isna().all():
            works.loc[mask, "cited_by_count"] = paper.get("citationCount")
            works.loc[mask, "source"] = works.loc[mask, "source"] + ";semantic_scholar"
        interim_rows.append(
            {
                "work_id": work_id,
                "s2_paper_id": paper.get("paperId"),
                "citation_count": paper.get("citationCount"),
                "external_ids": paper.get("externalIds") or {},
                "retrieved_at": retrieved_at,
                "source": "semantic_scholar",
            }
        )
        _map_author_ids(authorships, work_id, paper.get("authors") or [])
    LOG.info("stage=semantic_scholar works=%d", len(results))
    return works, authorships, pd.DataFrame(interim_rows, columns=interim_columns)


def _map_author_ids(
    authorships: pd.DataFrame,
    work_id: str,
    s2_authors: Sequence[Dict[str, Any]],
) -> None:
    work_rows = authorships[authorships["work_id"] == work_id]
    if work_rows.empty:
        return
    unique_openalex = work_rows.drop_duplicates("author_openalex_id")
    name_to_openalex: Dict[str, List[str]] = {}
    for row in unique_openalex.itertuples():
        name_to_openalex.setdefault(normalize_text(row.author_display_name), []).append(
            row.author_openalex_id
        )
    for s2_author in s2_authors:
        name = normalize_text(s2_author.get("name"))
        matches = name_to_openalex.get(name, [])
        if len(matches) != 1 or not s2_author.get("authorId"):
            continue
        mask = (
            (authorships["work_id"] == work_id)
            & (authorships["author_openalex_id"] == matches[0])
        )
        authorships.loc[mask, "author_s2_id"] = str(s2_author["authorId"])


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "semantic_scholar",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate["semantic_scholar_seconds"],
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={
            "Accept": "application/json",
            "x-api-key": cfg["_s2_api_key"],
        },
    )


def _chunks(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])
