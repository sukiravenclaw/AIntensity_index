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


# ---------------------------------------------------------------------------
# DBLP-native collection (OpenAlex is off). These supersede ``enrich`` and are
# wired into the pipeline in place of it.
# ---------------------------------------------------------------------------

PAPER_CROSSWALK_COLUMNS = [
    "work_id",
    "s2_paper_id",
    "citation_count",
    "retrieved_at",
    "source",
]

AUTHOR_METRICS_COLUMNS = [
    "author_dblp_pid",
    "s2_author_id",
    "s2_name",
    "citation_count",
    "h_index",
    "paper_count",
    "affiliations",
    "retrieved_at",
    "source",
]


def _work_id_col(frame: pd.DataFrame) -> str:
    return "work_id" if "work_id" in frame.columns else "dblp_work_id"


def _name_col(frame: pd.DataFrame) -> str:
    return "author_name" if "author_name" in frame.columns else "author_display_name"


def collect_citations(
    cfg: Dict[str, Any],
    works: pd.DataFrame,
    authorships: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DBLP-native S2 pass.

    Fills paper ``cited_by_count`` (by DOI) and returns author-level citation
    metrics keyed to DBLP PIDs (resolved via per-paper name match), plus a paper
    crosswalk. Returns ``(works, author_metrics, paper_crosswalk)``.
    """
    empty_authors = pd.DataFrame(columns=AUTHOR_METRICS_COLUMNS)
    empty_crosswalk = pd.DataFrame(columns=PAPER_CROSSWALK_COLUMNS)
    if works.empty or not cfg["semantic_scholar"].get("enabled", True):
        return works, empty_authors, empty_crosswalk
    if not cfg.get("_s2_api_key"):
        LOG.warning("stage=semantic_scholar skipped=true reason=missing_api_key")
        return works, empty_authors, empty_crosswalk

    wid = _work_id_col(works)
    name_col = _name_col(authorships)
    works = works.copy()
    if "cited_by_count" not in works.columns:
        works["cited_by_count"] = pd.NA
    doi_to_work: Dict[str, str] = {}
    for row in works.itertuples():
        doi = getattr(row, "doi", None)
        if doi:
            doi_to_work[f"DOI:{doi}"] = getattr(row, wid)

    client = _client(cfg)
    batch_size = min(500, int(cfg["semantic_scholar"].get("batch_size", 500)))

    works = works.copy()
    crosswalk_rows: List[Dict[str, Any]] = []
    # work_id -> list of {authorId, name} from S2, to resolve DBLP PIDs.
    work_s2_authors: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in _chunks(list(doi_to_work), batch_size):
        raw = client.request(
            "POST",
            f"{cfg['semantic_scholar']['endpoint'].rstrip('/')}/paper/batch",
            params={"fields": "externalIds,citationCount,authors"},
            json_body={"ids": chunk},
        )
        payload = raw.json()
        for requested, paper in zip(chunk, payload):
            if not paper:
                continue
            work_id = doi_to_work[requested]
            mask = works[wid] == work_id
            if works.loc[mask, "cited_by_count"].isna().all():
                works.loc[mask, "cited_by_count"] = paper.get("citationCount")
            crosswalk_rows.append(
                {
                    "work_id": work_id,
                    "s2_paper_id": paper.get("paperId"),
                    "citation_count": paper.get("citationCount"),
                    "retrieved_at": raw.retrieved_at,
                    "source": "semantic_scholar",
                }
            )
            work_s2_authors[work_id] = paper.get("authors") or []

    # Resolve S2 author IDs -> DBLP PIDs by within-work name match.
    s2id_to_pid = _resolve_author_ids(authorships, work_s2_authors, wid, name_col)
    author_metrics = _collect_author_metrics(cfg, client, s2id_to_pid)

    paper_crosswalk = pd.DataFrame(crosswalk_rows, columns=PAPER_CROSSWALK_COLUMNS)
    LOG.info(
        "stage=semantic_scholar papers=%d authors_resolved=%d",
        len(paper_crosswalk),
        len(s2id_to_pid),
    )
    return works, author_metrics, paper_crosswalk


def _resolve_author_ids(
    authorships: pd.DataFrame,
    work_s2_authors: Dict[str, List[Dict[str, Any]]],
    wid: str,
    name_col: str,
) -> Dict[str, str]:
    """Map S2 authorId -> DBLP PID using unambiguous within-work name matches."""
    s2id_to_pid: Dict[str, str] = {}
    if "author_dblp_pid" not in authorships.columns:
        return s2id_to_pid
    by_work = authorships.groupby(wid)
    for work_id, s2_authors in work_s2_authors.items():
        if work_id not in by_work.groups:
            continue
        rows = by_work.get_group(work_id)
        name_to_pid: Dict[str, List[str]] = {}
        for row in rows.itertuples():
            nm = normalize_text(getattr(row, name_col, None))
            if nm:
                name_to_pid.setdefault(nm, []).append(row.author_dblp_pid)
        for s2_author in s2_authors:
            nm = normalize_text(s2_author.get("name"))
            pids = name_to_pid.get(nm, [])
            author_id = s2_author.get("authorId")
            if author_id and len(set(pids)) == 1:
                s2id_to_pid[str(author_id)] = pids[0]
    return s2id_to_pid


def _collect_author_metrics(
    cfg: Dict[str, Any],
    client: RawCacheClient,
    s2id_to_pid: Dict[str, str],
) -> pd.DataFrame:
    if not s2id_to_pid:
        return pd.DataFrame(columns=AUTHOR_METRICS_COLUMNS)
    rows: List[Dict[str, Any]] = []
    batch_size = min(1000, int(cfg["semantic_scholar"].get("batch_size", 500)))
    ids = list(s2id_to_pid)
    for chunk in _chunks(ids, batch_size):
        raw = client.request(
            "POST",
            f"{cfg['semantic_scholar']['endpoint'].rstrip('/')}/author/batch",
            params={"fields": "name,affiliations,paperCount,citationCount,hIndex"},
            json_body={"ids": chunk},
        )
        payload = raw.json()
        for requested, author in zip(chunk, payload):
            if not author:
                continue
            rows.append(
                {
                    "author_dblp_pid": s2id_to_pid[requested],
                    "s2_author_id": author.get("authorId") or requested,
                    "s2_name": author.get("name"),
                    "citation_count": author.get("citationCount"),
                    "h_index": author.get("hIndex"),
                    "paper_count": author.get("paperCount"),
                    "affiliations": "|".join(author.get("affiliations") or []),
                    "retrieved_at": raw.retrieved_at,
                    "source": "semantic_scholar",
                }
            )
    return pd.DataFrame(rows, columns=AUTHOR_METRICS_COLUMNS)


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
