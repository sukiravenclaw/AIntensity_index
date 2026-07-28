from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.dblp")

WORKS_COLUMNS = [
    "dblp_work_id",
    "title",
    "publication_year",
    "doi",
    "venue_normalized",
    "venue_raw",
    "retrieved_at",
    "source",
]

AUTHORSHIPS_COLUMNS = [
    "dblp_work_id",
    "author_dblp_pid",
    "author_name",
    "author_position",
    "author_ordinal",
    "retrieved_at",
    "source",
]

# QLever (dblp.org SPARQL) streams large results, but we page defensively so a
# venue-year sweep enriched with per-signature author rows cannot silently hit a
# server-side row cap.
_PAGE_SIZE = 20000


def collect(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Backward-compatible works-only seed (unchanged public contract)."""
    works, _ = collect_authorships(cfg)
    return works


def collect_authorships(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Venue-year seed plus the per-work author (signature) rows.

    Returns ``(works, authorships)``. ``authorships`` carries the DBLP person id
    (``author_dblp_pid``), the signature name, and a derived ``author_position``
    (first / middle / last) from the signature ordinal.
    """
    if not cfg["dblp"].get("enabled", True):
        return pd.DataFrame(columns=WORKS_COLUMNS), pd.DataFrame(columns=AUTHORSHIPS_COLUMNS)

    streams = {
        details["dblp_stream"]: venue
        for venue, details in cfg["venues"].items()
        if details.get("dblp_stream")
    }
    values = " ".join(f"<https://dblp.org/streams/{stream}>" for stream in streams)
    start_year = cfg["window"]["start"][:4]
    end_year = cfg["window"]["end"][:4]
    client = _client(cfg)

    work_rows: Dict[str, Dict[str, Any]] = {}
    signature_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = _build_query(values, start_year, end_year, _PAGE_SIZE, offset)
        raw = client.request(
            "GET",
            cfg["dblp"]["endpoint"],
            params={"query": query, "format": "application/sparql-results+json"},
        )
        bindings = raw.json().get("results", {}).get("bindings", [])
        for binding in bindings:
            stream_uri = _value(binding, "stream") or ""
            stream = stream_uri.split("/streams/", 1)[-1]
            venue = streams.get(stream)
            if not venue:
                continue
            work_id = _value(binding, "work")
            if not work_id:
                continue
            if work_id not in work_rows:
                work_rows[work_id] = {
                    "dblp_work_id": work_id,
                    "title": _value(binding, "title"),
                    "publication_year": _safe_year(_value(binding, "year")),
                    "doi": _normalize_doi(_value(binding, "doi")),
                    "venue_normalized": venue,
                    "venue_raw": _value(binding, "publishedIn") or venue,
                    "retrieved_at": raw.retrieved_at,
                    "source": "dblp",
                }
            person_uri = _value(binding, "person")
            if person_uri:
                signature_rows.append(
                    {
                        "dblp_work_id": work_id,
                        "author_dblp_pid": _pid(person_uri),
                        "author_name": _value(binding, "name"),
                        "author_ordinal": _safe_int(_value(binding, "ord")),
                        "retrieved_at": raw.retrieved_at,
                        "source": "dblp",
                    }
                )
        if len(bindings) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    works = pd.DataFrame(list(work_rows.values()), columns=WORKS_COLUMNS)
    authorships = _finalize_authorships(signature_rows)
    LOG.info("stage=dblp works=%d authorships=%d", len(works), len(authorships))
    return works, authorships


def _finalize_authorships(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=AUTHORSHIPS_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.dropna(subset=["author_dblp_pid"]).drop_duplicates(
        ["dblp_work_id", "author_dblp_pid"]
    )
    # Derive first/middle/last from the ordinal within each work.
    max_ord = frame.groupby("dblp_work_id")["author_ordinal"].transform("max")
    position = pd.Series("middle", index=frame.index)
    position = position.mask(frame["author_ordinal"] == 1, "first")
    position = position.mask(
        (frame["author_ordinal"] == max_ord) & (max_ord > 1), "last"
    )
    # A single-author work is both first and last; label it "first".
    frame["author_position"] = position
    return frame[AUTHORSHIPS_COLUMNS]


def _build_query(values: str, start_year: str, end_year: str, limit: int, offset: int) -> str:
    return f"""
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?work ?title ?year ?doi ?stream ?publishedIn ?person ?name ?ord WHERE {{
  VALUES ?stream {{ {values} }}
  ?work dblp:publishedInStream ?stream ;
        dblp:title ?title ;
        dblp:yearOfPublication ?year ;
        dblp:hasSignature ?sig .
  ?sig dblp:signatureCreator ?person .
  OPTIONAL {{ ?sig dblp:signatureOrdinal ?ord . }}
  OPTIONAL {{ ?sig dblp:signatureDblpName ?name . }}
  OPTIONAL {{ ?work dblp:doi ?doi . }}
  OPTIONAL {{ ?work dblp:publishedIn ?publishedIn . }}
  FILTER(?year >= "{start_year}"^^xsd:gYear && ?year <= "{end_year}"^^xsd:gYear)
}}
ORDER BY ?work ?ord
LIMIT {limit} OFFSET {offset}
""".strip()


def _pid(person_uri: str) -> str | None:
    if not person_uri:
        return None
    if "/pid/" in person_uri:
        return person_uri.split("/pid/", 1)[-1].strip("/")
    return person_uri.rstrip("/").rsplit("/", 1)[-1] or None


def _value(binding: Dict[str, Any], key: str) -> Any:
    return binding.get(key, {}).get("value")


def _safe_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_year(value: Any) -> Any:
    year = _safe_int(value)
    return year


def _normalize_doi(value: Any) -> Any:
    if not value:
        return None
    text = str(value).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.lower()


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "dblp",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate["dblp_seconds"],
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/sparql-results+json"},
    )
