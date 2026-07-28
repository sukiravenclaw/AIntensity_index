"""Free founder / employer / education overlay from Wikidata SPARQL.

High precision, low recall — used only to overlay high-confidence facts on top
of the DBLP/S2 spine. Two entry points:

- ``collect_founders`` — for our curated AI companies, who founded them, with the
  founder's DBLP author id (P2456) and ORCID (P496) for a clean person join. This
  populates the required ``founder`` column (e.g. Zhilin Yang -> Moonshot).
- ``collect_person_facts`` — for people with a known DBLP author id, their
  employer (P108) and educated-at (P69). Overlay for the employer cascade.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.common import RawCacheClient


LOG = logging.getLogger("ai_talent_collection.wikidata")

FOUNDER_COLUMNS = [
    "org_id",
    "company_label",
    "founder_name",
    "founder_dblp_pid",
    "founder_orcid",
    "founder_qid",
    "retrieved_at",
    "source",
]

PERSON_FACT_COLUMNS = [
    "author_dblp_pid",
    "fact_type",  # "employer" | "educated_at"
    "org_label",
    "retrieved_at",
    "source",
]

_LABEL_CHUNK = 60
_PID_CHUNK = 80


def collect_founders(
    cfg: Dict[str, Any],
    company_label_to_org_id: Dict[str, str],
) -> pd.DataFrame:
    if not cfg.get("wikidata", {}).get("enabled", True) or not company_label_to_org_id:
        return pd.DataFrame(columns=FOUNDER_COLUMNS)

    client = _client(cfg)
    endpoint = cfg["wikidata"]["endpoint"]
    labels = sorted(company_label_to_org_id)
    rows: List[Dict[str, Any]] = []
    for chunk in _chunks(labels, _LABEL_CHUNK):
        query = _founders_query(chunk)
        try:
            raw = client.request("GET", endpoint, params={"query": query, "format": "json"})
        except Exception as exc:
            LOG.warning("stage=wikidata founders_chunk_failed=%s", exc)
            continue
        for b in raw.json().get("results", {}).get("bindings", []):
            label = _v(b, "companyLabel")
            org_id = company_label_to_org_id.get(label)
            if not org_id:
                continue
            rows.append(
                {
                    "org_id": org_id,
                    "company_label": label,
                    "founder_name": _v(b, "founderLabel"),
                    "founder_dblp_pid": _v(b, "dblp"),
                    "founder_orcid": _v(b, "orcid"),
                    "founder_qid": _qid(_v(b, "founder")),
                    "retrieved_at": raw.retrieved_at,
                    "source": "wikidata",
                }
            )
    frame = pd.DataFrame(rows, columns=FOUNDER_COLUMNS).drop_duplicates(
        ["org_id", "founder_qid"]
    )
    LOG.info("stage=wikidata founders=%d companies=%d", len(frame), len(company_label_to_org_id))
    return frame


def collect_person_facts(cfg: Dict[str, Any], dblp_pids: Iterable[str]) -> pd.DataFrame:
    if not cfg.get("wikidata", {}).get("enabled", True):
        return pd.DataFrame(columns=PERSON_FACT_COLUMNS)
    pids = sorted({p for p in dblp_pids if p})
    if not pids:
        return pd.DataFrame(columns=PERSON_FACT_COLUMNS)

    client = _client(cfg)
    endpoint = cfg["wikidata"]["endpoint"]
    rows: List[Dict[str, Any]] = []
    for chunk in _chunks(pids, _PID_CHUNK):
        query = _person_facts_query(chunk)
        try:
            raw = client.request("GET", endpoint, params={"query": query, "format": "json"})
        except Exception as exc:
            LOG.warning("stage=wikidata person_facts_chunk_failed=%s", exc)
            continue
        for b in raw.json().get("results", {}).get("bindings", []):
            pid = _v(b, "dblp")
            if _v(b, "employerLabel"):
                rows.append({"author_dblp_pid": pid, "fact_type": "employer",
                             "org_label": _v(b, "employerLabel"),
                             "retrieved_at": raw.retrieved_at, "source": "wikidata"})
            if _v(b, "eduLabel"):
                rows.append({"author_dblp_pid": pid, "fact_type": "educated_at",
                             "org_label": _v(b, "eduLabel"),
                             "retrieved_at": raw.retrieved_at, "source": "wikidata"})
    frame = pd.DataFrame(rows, columns=PERSON_FACT_COLUMNS).drop_duplicates()
    LOG.info("stage=wikidata person_facts=%d pids=%d", len(frame), len(pids))
    return frame


def _founders_query(labels: List[str]) -> str:
    values = " ".join(f'"{_escape(l)}"@en' for l in labels)
    return f"""
SELECT ?companyLabel ?founder ?founderLabel ?dblp ?orcid WHERE {{
  VALUES ?name {{ {values} }}
  ?company rdfs:label ?name .
  ?company wdt:P112 ?founder .
  OPTIONAL {{ ?founder wdt:P2456 ?dblp . }}
  OPTIONAL {{ ?founder wdt:P496 ?orcid . }}
  ?founder rdfs:label ?founderLabel . FILTER(LANG(?founderLabel) = "en")
  BIND(?name AS ?companyLabel)
}}
""".strip()


def _person_facts_query(pids: List[str]) -> str:
    values = " ".join(f'"{_escape(p)}"' for p in pids)
    return f"""
SELECT ?dblp ?employerLabel ?eduLabel WHERE {{
  VALUES ?dblp {{ {values} }}
  ?person wdt:P2456 ?dblp .
  OPTIONAL {{ ?person wdt:P108 ?employer . ?employer rdfs:label ?employerLabel . FILTER(LANG(?employerLabel)="en") }}
  OPTIONAL {{ ?person wdt:P69 ?edu . ?edu rdfs:label ?eduLabel . FILTER(LANG(?eduLabel)="en") }}
}}
""".strip()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _v(binding: Dict[str, Any], key: str) -> Any:
    return binding.get(key, {}).get("value")


def _qid(uri: Any) -> Any:
    if not uri:
        return None
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _chunks(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _client(cfg: Dict[str, Any]) -> RawCacheClient:
    rate = cfg["rate_limits"]
    return RawCacheClient(
        "wikidata",
        cfg["paths"]["raw"],
        cfg["_user_agent"],
        rate.get("wikidata_seconds", 1.5),
        rate["timeout_seconds"],
        rate["max_retries"],
        headers={"Accept": "application/sparql-results+json"},
    )
