from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.dblp")


def collect(cfg: Dict[str, Any]) -> pd.DataFrame:
    columns = [
        "dblp_work_id",
        "title",
        "publication_year",
        "doi",
        "venue_normalized",
        "venue_raw",
        "retrieved_at",
        "source",
    ]
    if not cfg["dblp"].get("enabled", True):
        return pd.DataFrame(columns=columns)

    streams = {
        details["dblp_stream"]: venue
        for venue, details in cfg["venues"].items()
        if details.get("dblp_stream")
    }
    values = " ".join(f"<https://dblp.org/streams/{stream}>" for stream in streams)
    start_year = cfg["window"]["start"][:4]
    end_year = cfg["window"]["end"][:4]
    query = f"""
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?work ?title ?year ?doi ?stream ?publishedIn WHERE {{
  VALUES ?stream {{ {values} }}
  ?work dblp:publishedInStream ?stream ;
        dblp:title ?title ;
        dblp:yearOfPublication ?year .
  OPTIONAL {{ ?work dblp:doi ?doi . }}
  OPTIONAL {{ ?work dblp:publishedIn ?publishedIn . }}
  FILTER(?year >= "{start_year}"^^xsd:gYear && ?year <= "{end_year}"^^xsd:gYear)
}}
ORDER BY ?year ?work
""".strip()
    client = _client(cfg)
    raw = client.request(
        "GET",
        cfg["dblp"]["endpoint"],
        params={"query": query, "format": "application/sparql-results+json"},
    )
    payload = raw.json()
    rows: List[Dict[str, Any]] = []
    for binding in payload.get("results", {}).get("bindings", []):
        stream_uri = _value(binding, "stream")
        stream = stream_uri.split("/streams/", 1)[-1]
        venue = streams.get(stream)
        if not venue:
            continue
        doi = _value(binding, "doi")
        rows.append(
            {
                "dblp_work_id": _value(binding, "work"),
                "title": _value(binding, "title"),
                "publication_year": int(_value(binding, "year")),
                "doi": _normalize_doi(doi),
                "venue_normalized": venue,
                "venue_raw": _value(binding, "publishedIn") or venue,
                "retrieved_at": raw.retrieved_at,
                "source": "dblp",
            }
        )
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates("dblp_work_id")
    LOG.info("stage=dblp rows=%d", len(frame))
    return frame


def _value(binding: Dict[str, Any], key: str) -> Any:
    return binding.get(key, {}).get("value")


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

