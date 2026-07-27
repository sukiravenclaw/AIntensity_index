from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.common import RawCacheClient
from src.normalize.text import normalize_doi, normalize_text, normalize_title


LOG = logging.getLogger("ai_talent_collection.openalex")
ARXIV_RE = re.compile(r"(?:arxiv(?:\.org)?[/:\s]+(?:abs/)?)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def collect(
    cfg: Dict[str, Any],
    dblp: pd.DataFrame,
    openreview: pd.DataFrame,
    arxiv: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve scoped source records to OpenAlex works and authorships."""
    seeds = _build_seeds(cfg, dblp, openreview, arxiv)
    seed_indexes = _seed_indexes(seeds)
    client = _client(cfg)
    candidates: Dict[str, Dict[str, Any]] = {}

    # DOI bulk resolution is high precision and inexpensive.
    dois = sorted({value for value in seeds["doi"].dropna().tolist() if value})
    for chunk in _chunks(dois, 50):
        raw = client.request(
            "GET",
            f"{cfg['openalex']['endpoint'].rstrip('/')}/works",
            params=_params(
                cfg,
                {
                    "filter": "doi:" + "|".join(chunk),
                    "per_page": min(100, len(chunk)),
                },
            ),
        )
        for work in raw.json().get("results", []):
            _remember(candidates, work, raw.retrieved_at)

    # A source sweep amortizes title-only matching, especially for arXiv and
    # proceedings that do not assign DOIs.
    if cfg["openalex"].get("source_sweep", True):
        source_ids = _resolve_source_ids(cfg, client)
        if source_ids:
            for work, retrieved_at in _list_source_works(cfg, client, source_ids):
                if _candidate_seed_indices(work, seed_indexes):
                    _remember(candidates, work, retrieved_at)

    matched_seed_indices = _matched_seed_indices(
        seeds,
        seed_indexes,
        candidates.values(),
        cfg,
    )
    unmatched = seeds.loc[~seeds.index.isin(matched_seed_indices)]
    LOG.info(
        "stage=openalex initial_candidates=%d unmatched_seeds=%d",
        len(candidates),
        len(unmatched),
    )

    # Remaining records are resolved individually by exact/near-exact title.
    # Skip if disabled in config to avoid rate limits
    if cfg["openalex"].get("enable_title_search", True):
        for row in unmatched.itertuples():
            if not row.title:
                continue
            filters = (
                f"from_publication_date:{cfg['window']['start']},"
                f"to_publication_date:{cfg['window']['end']}"
            )
            try:
                raw = client.request(
                    "GET",
                    f"{cfg['openalex']['endpoint'].rstrip('/')}/works",
                    params=_params(
                        cfg,
                        {
                            "search": row.title,
                            "filter": filters,
                            "per_page": 5,
                        },
                    ),
                )
                best = _best_title_match(
                    row.title,
                    row.publication_year,
                    raw.json().get("results", []),
                    float(cfg["openalex"].get("title_match_threshold", 0.985)),
                )
                if best is not None:
                    _remember(candidates, best, raw.retrieved_at, seed_indices={row.Index})
            except Exception as exc:
                # Skip papers that cause API errors (e.g., titles with special characters)
                LOG.warning(
                    "stage=openalex_title_search_failed title=%r error=%s",
                    row.title[:80],
                    str(exc),
                )
                continue
    else:
        LOG.info(
            "stage=openalex_title_search_skipped unmatched=%d reason=disabled_in_config",
            len(unmatched),
        )

    works_rows: List[Dict[str, Any]] = []
    authorship_rows: List[Dict[str, Any]] = []
    for entry in candidates.values():
        work = entry["work"]
        seed = _best_seed_for_work(
            work,
            seeds,
            seed_indexes,
            cfg,
            entry.get("seed_indices"),
        )
        if seed is None:
            continue
        work_row, author_rows = _parse_work(
            work,
            seed,
            entry["retrieved_at"],
            cfg,
        )
        works_rows.append(work_row)
        authorship_rows.extend(author_rows)

    works = pd.DataFrame(works_rows)
    authorships = pd.DataFrame(authorship_rows)
    if not works.empty:
        works = works.drop_duplicates("work_id").sort_values(
            ["publication_year", "venue_normalized", "work_id"]
        )
    if not authorships.empty:
        authorships = authorships.drop_duplicates(
            [
                "work_id",
                "author_openalex_id",
                "institution_ror",
                "institution_display_name",
                "raw_affiliation_string",
            ]
        )
    LOG.info("stage=openalex works=%d authorships=%d", len(works), len(authorships))
    return works, authorships


def _build_seeds(
    cfg: Dict[str, Any],
    dblp: pd.DataFrame,
    openreview: pd.DataFrame,
    arxiv: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for frame in (dblp, openreview):
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "doi": normalize_doi(row.get("doi")),
                    "arxiv_id": row.get("arxiv_id"),
                    "openreview_id": row.get("openreview_id"),
                    "title": row.get("title"),
                    "title_normalized": normalize_title(row.get("title")),
                    "publication_year": row.get("publication_year"),
                    "venue_normalized": row.get("venue_normalized"),
                    "venue_raw": row.get("venue_raw"),
                    "decision": row.get("decision") or "unknown",
                    "seed_source": row.get("source"),
                }
            )
    for row in arxiv.to_dict("records"):
        rows.append(
            {
                "doi": normalize_doi(row.get("doi")),
                "arxiv_id": row.get("arxiv_id"),
                "openreview_id": None,
                "title": row.get("title"),
                "title_normalized": normalize_title(row.get("title")),
                "publication_year": row.get("publication_year"),
                "venue_normalized": "arXiv",
                "venue_raw": "arXiv " + str(row.get("primary_category") or ""),
                "decision": "preprint",
                "seed_source": "arxiv",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "doi",
                "arxiv_id",
                "openreview_id",
                "title",
                "title_normalized",
                "publication_year",
                "venue_normalized",
                "venue_raw",
                "decision",
                "seed_source",
            ]
        )
    seeds = pd.DataFrame(rows)
    # Keep all source claims; precedence is applied after matching.
    return seeds.drop_duplicates(
        ["doi", "arxiv_id", "openreview_id", "title_normalized", "venue_normalized"]
    ).reset_index(drop=True)


def _resolve_source_ids(cfg: Dict[str, Any], client: RawCacheClient) -> Dict[str, Set[str]]:
    source_ids: Dict[str, Set[str]] = {}
    queries: List[Tuple[str, List[str]]] = [
        ("arXiv", ["arXiv"]),
    ]
    for venue, details in cfg["venues"].items():
        queries.append((venue, list(details.get("aliases") or [venue])))
    for label, aliases in queries:
        selected: Set[str] = set()
        for query in aliases[:2]:
            raw = client.request(
                "GET",
                f"{cfg['openalex']['endpoint'].rstrip('/')}/sources",
                params=_params(cfg, {"search": query, "per_page": 20}),
            )
            for source in raw.json().get("results", []):
                name = normalize_text(source.get("display_name"))
                if _source_name_matches(name, aliases):
                    identifier = source.get("id")
                    if identifier:
                        selected.add(identifier.rsplit("/", 1)[-1])
        if selected:
            source_ids[label] = selected
            LOG.info("stage=openalex_source_resolution label=%s ids=%s", label, sorted(selected))
    return source_ids


def _source_name_matches(name: str, aliases: Sequence[str]) -> bool:
    if not name:
        return False
    for alias in aliases:
        candidate = normalize_text(alias)
        if candidate and (candidate == name or candidate in name or name in candidate):
            return True
    return False


def _list_source_works(
    cfg: Dict[str, Any],
    client: RawCacheClient,
    source_ids: Dict[str, Set[str]],
) -> Iterable[Tuple[Dict[str, Any], str]]:
    all_ids = sorted({identifier for values in source_ids.values() for identifier in values})
    max_pages = cfg["openalex"].get("max_pages_per_source")
    for id_chunk in _chunks(all_ids, 50):
        cursor = "*"
        page = 0
        while cursor:
            if max_pages is not None and page >= int(max_pages):
                break
            filters = (
                "locations.source.id:" + "|".join(id_chunk)
                + f",from_publication_date:{cfg['window']['start']}"
                + f",to_publication_date:{cfg['window']['end']}"
            )
            raw = client.request(
                "GET",
                f"{cfg['openalex']['endpoint'].rstrip('/')}/works",
                params=_params(
                    cfg,
                    {
                        "filter": filters,
                        "per_page": int(cfg["openalex"].get("page_size", 100)),
                        "cursor": cursor,
                    },
                ),
            )
            payload = raw.json()
            results = payload.get("results", [])
            for work in results:
                yield work, raw.retrieved_at
            cursor = (payload.get("meta") or {}).get("next_cursor")
            page += 1
            LOG.info(
                "stage=openalex_source_sweep page=%d rows=%d cursor=%s",
                page,
                len(results),
                bool(cursor),
            )
            if not results:
                break


def _seed_indexes(seeds: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    indexes: Dict[str, Dict[str, List[int]]] = {
        "doi": {},
        "arxiv_id": {},
        "title_normalized": {},
    }
    for index, seed in seeds.iterrows():
        for column in indexes:
            value = seed.get(column)
            if value is not None and not pd.isna(value) and str(value):
                indexes[column].setdefault(str(value), []).append(index)
    return indexes


def _candidate_seed_indices(
    work: Dict[str, Any],
    indexes: Dict[str, Dict[str, List[int]]],
) -> Set[int]:
    candidates: Set[int] = set()
    values = {
        "doi": normalize_doi(work.get("doi")),
        "arxiv_id": _extract_work_arxiv_id(work),
        "title_normalized": normalize_title(work.get("title") or work.get("display_name")),
    }
    for column, value in values.items():
        if value:
            candidates.update(indexes[column].get(str(value), []))
    return candidates


def _matched_seed_indices(
    seeds: pd.DataFrame,
    indexes: Dict[str, Dict[str, List[int]]],
    entries: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Set[int]:
    matched: Set[int] = set()
    for entry in entries:
        work = entry.get("work", entry)
        for index in _candidate_seed_indices(work, indexes):
            seed = seeds.loc[index]
            if _seed_match_score(work, seed, cfg) > 0:
                matched.add(index)
    return matched


def _best_seed_for_work(
    work: Dict[str, Any],
    seeds: pd.DataFrame,
    indexes: Dict[str, Dict[str, List[int]]],
    cfg: Dict[str, Any],
    forced_indices: Optional[Set[int]] = None,
) -> Optional[pd.Series]:
    scored: List[Tuple[float, int, pd.Series]] = []
    indices = _candidate_seed_indices(work, indexes).union(forced_indices or set())
    for index in indices:
        seed = seeds.loc[index]
        score = _seed_match_score(work, seed, cfg)
        if score <= 0:
            continue
        # Prefer conference claims to preprint claims for a single publication.
        venue_priority = 0 if seed.get("venue_normalized") == "arXiv" else 10
        source_priority = {"openreview": 3, "dblp": 2, "arxiv": 1}.get(seed.get("seed_source"), 0)
        scored.append((score, venue_priority + source_priority, seed))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _seed_match_score(work: Dict[str, Any], seed: pd.Series, cfg: Dict[str, Any]) -> float:
    work_doi = normalize_doi(work.get("doi"))
    if work_doi and seed.get("doi") and work_doi == seed.get("doi"):
        return 4.0
    work_arxiv = _extract_work_arxiv_id(work)
    if work_arxiv and seed.get("arxiv_id") and work_arxiv == seed.get("arxiv_id"):
        return 3.5
    left = normalize_title(work.get("title") or work.get("display_name"))
    right = seed.get("title_normalized") or ""
    if not left or not right:
        return 0.0
    work_year = work.get("publication_year")
    seed_year = seed.get("publication_year")
    if work_year and seed_year and abs(int(work_year) - int(seed_year)) > 1:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    threshold = float(cfg["openalex"].get("title_match_threshold", 0.985))
    return 2.0 + ratio if ratio >= threshold else 0.0


def _best_title_match(
    title: str,
    year: Any,
    works: Sequence[Dict[str, Any]],
    threshold: float,
) -> Optional[Dict[str, Any]]:
    target = normalize_title(title)
    best: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
    for work in works:
        candidate_year = work.get("publication_year")
        if year and candidate_year and abs(int(year) - int(candidate_year)) > 1:
            continue
        ratio = SequenceMatcher(
            None,
            target,
            normalize_title(work.get("title") or work.get("display_name")),
        ).ratio()
        if ratio > best[0]:
            best = (ratio, work)
    return best[1] if best[0] >= threshold else None


def _parse_work(
    work: Dict[str, Any],
    seed: pd.Series,
    retrieved_at: str,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    work_id = work.get("id")
    authorships = work.get("authorships") or []
    n_authors = len(authorships)
    primary_location = work.get("primary_location") or {}
    source_obj = primary_location.get("source") or {}
    metric = work.get("citation_normalized_percentile")
    metric_value = metric.get("value") if isinstance(metric, dict) else metric
    arxiv_id = seed.get("arxiv_id") or _extract_work_arxiv_id(work)
    venue = seed.get("venue_normalized")
    tier = "preprint" if venue == "arXiv" else cfg["venues"].get(venue, {}).get("tier")
    topics = [
        topic.get("display_name")
        for topic in (work.get("topics") or [])
        if topic.get("display_name")
    ]
    primary_topic = (work.get("primary_topic") or {}).get("display_name")
    work_row = {
        "work_id": work_id,
        "doi": normalize_doi(work.get("doi")),
        "arxiv_id": arxiv_id,
        "openreview_id": seed.get("openreview_id"),
        "s2_paper_id": None,
        "title": work.get("title") or work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "venue_raw": seed.get("venue_raw") or source_obj.get("display_name"),
        "venue_normalized": venue,
        "venue_tier": tier,
        "decision": seed.get("decision") or ("preprint" if venue == "arXiv" else "unknown"),
        "cited_by_count": work.get("cited_by_count"),
        "fwci": work.get("fwci"),
        "citation_normalized_percentile": metric_value,
        "topics": topics,
        "primary_topic": primary_topic,
        "n_authors": n_authors,
        "retrieved_at": retrieved_at,
        "source": "openalex",
        "_fwci_field_present": "fwci" in work,
        "_percentile_field_present": "citation_normalized_percentile" in work,
    }
    author_rows: List[Dict[str, Any]] = []
    for position, authorship in enumerate(authorships):
        author = authorship.get("author") or {}
        institutions = authorship.get("institutions") or []
        raw_strings = authorship.get("raw_affiliation_strings") or []
        if isinstance(raw_strings, str):
            raw_strings = [raw_strings]
        row_count = max(len(institutions), len(raw_strings), 1)
        n_institutions = max(len(institutions), len(raw_strings), 1)
        for index in range(row_count):
            institution = institutions[index] if index < len(institutions) else {}
            if len(raw_strings) == 1:
                raw_affiliation = raw_strings[0]
            elif index < len(raw_strings):
                raw_affiliation = raw_strings[index]
            else:
                raw_affiliation = None
            # Fractional credit: 1 / n_institutions to avoid double-counting
            # when same author has multiple institutions on one paper
            fractional_credit = 1.0 / n_institutions if n_institutions > 0 else 1.0
            author_rows.append(
                {
                    "work_id": work_id,
                    "author_openalex_id": author.get("id"),
                    "author_s2_id": None,
                    "orcid": author.get("orcid"),
                    "author_display_name": author.get("display_name") or authorship.get("raw_author_name"),
                    "author_position": authorship.get("author_position")
                    or _author_position(position, n_authors),
                    "n_authors": n_authors,
                    "n_institutions": n_institutions,
                    "fractional_credit": fractional_credit,
                    "raw_affiliation_string": raw_affiliation,
                    "institution_ror": institution.get("ror"),
                    "institution_display_name": institution.get("display_name"),
                    "institution_type": institution.get("type"),
                    "institution_country": institution.get("country_code"),
                    "org_id": None,
                    "retrieved_at": retrieved_at,
                    "source": "openalex",
                }
            )
    return work_row, author_rows


def apply_metric_fallbacks(works: pd.DataFrame) -> pd.DataFrame:
    if works.empty:
        return works
    grouping = ["publication_year", "primary_topic"]
    citations = pd.to_numeric(works["cited_by_count"], errors="coerce")
    group_mean = citations.groupby([works[column] for column in grouping], dropna=False).transform("mean")
    local_fwci = citations / group_mean.replace(0, pd.NA)
    local_percentile = citations.groupby(
        [works[column] for column in grouping],
        dropna=False,
    ).rank(method="average", pct=True)
    if not bool(works["_fwci_field_present"].any()):
        LOG.warning("stage=metrics field=fwci fallback=local_year_topic")
        works["fwci"] = local_fwci
        works["source"] = works["source"] + ";local_year_topic_metrics"
    else:
        works["fwci"] = works["fwci"].where(works["fwci"].notna(), local_fwci)
    if not bool(works["_percentile_field_present"].any()):
        LOG.warning("stage=metrics field=citation_normalized_percentile fallback=local_year_topic")
        works["citation_normalized_percentile"] = local_percentile
        marker = ~works["source"].str.contains("local_year_topic_metrics", regex=False)
        works.loc[marker, "source"] = works.loc[marker, "source"] + ";local_year_topic_metrics"
    else:
        works["citation_normalized_percentile"] = works[
            "citation_normalized_percentile"
        ].where(works["citation_normalized_percentile"].notna(), local_percentile)
    return works.drop(columns=["_fwci_field_present", "_percentile_field_present"])


def _extract_work_arxiv_id(work: Dict[str, Any]) -> Optional[str]:
    locations = work.get("locations") or []
    for location in locations:
        for key in ("landing_page_url", "pdf_url"):
            value = location.get(key) or ""
            match = ARXIV_RE.search(value)
            if match:
                return match.group(1)
    return None


def _author_position(index: int, count: int) -> str:
    if index == 0:
        return "first"
    if index == count - 1:
        return "last"
    return "middle"


def _remember(
    candidates: Dict[str, Dict[str, Any]],
    work: Dict[str, Any],
    retrieved_at: str,
    seed_indices: Optional[Set[int]] = None,
) -> None:
    identifier = work.get("id")
    if identifier:
        current = candidates.get(identifier, {})
        combined = set(current.get("seed_indices") or set()).union(seed_indices or set())
        candidates[identifier] = {
            "work": work,
            "retrieved_at": retrieved_at,
            "seed_indices": combined,
        }


def _params(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(params)
    if cfg.get("_openalex_api_key"):
        output["api_key"] = cfg["_openalex_api_key"]
    return output


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "openalex",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate["openalex_seconds"],
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/json"},
    )


def _chunks(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])
