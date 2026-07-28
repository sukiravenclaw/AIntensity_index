"""Current-employer resolution cascade (OpenAlex-free).

No single free source gives reliable current employer, so we layer them
highest-precision-first, each contributing a confidence:

    wikidata (1.0) > founder (1.0) > orcid latest (0.9) > homepage domain (0.7)
    > s2 affiliation (0.4) > unknown

Employer *name* strings (Wikidata/ORCID/S2) are mapped to ``org_id`` via the
curated alias index (``src/normalize/orgs.build_indexes``). Homepage URLs use a
small corporate-domain map plus a generic academic-domain fallthrough.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from src.normalize.orgs import match_affiliation

LOG = logging.getLogger("ai_talent_collection.employer")

CONFIDENCE = {
    "wikidata": 1.0,
    "founder": 1.0,
    "orcid": 0.9,
    "homepage": 0.7,
    "s2": 0.4,
}

# Cascade order (highest precision first).
_ORDER = ["wikidata", "founder", "orcid", "homepage", "s2"]

# High-value corporate domains → org_id. Extend as needed; name-matching handles
# the long tail.
CORPORATE_DOMAIN_ORG = {
    "openai.com": "openai",
    "anthropic.com": "anthropic",
    "deepmind.com": "google_deepmind",
    "moonshot.cn": "moonshot",
    "moonshot.ai": "moonshot",
    "mistral.ai": "mistral",
    "x.ai": "xai",
    "cohere.com": "cohere",
    "cohere.ai": "cohere",
    "perplexity.ai": "perplexity",
    "together.ai": "together_ai",
    "stability.ai": "stability_ai",
}

EMPLOYER_COLUMNS = [
    "author_dblp_pid",
    "employer_org_id",
    "source",
    "confidence",
    "as_of_year",
]


def homepage_domain(url: Any) -> Optional[str]:
    if not url:
        return None
    text = str(url).strip()
    if "//" not in text:
        text = "//" + text
    host = (urlparse(text).hostname or "").lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def employer_from_homepage(url: Any, alias_map: Dict[str, str]) -> Optional[str]:
    host = homepage_domain(url)
    if not host:
        return None
    # Exact then registrable-suffix match against the corporate map.
    if host in CORPORATE_DOMAIN_ORG:
        return CORPORATE_DOMAIN_ORG[host]
    parts = host.split(".")
    for i in range(len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in CORPORATE_DOMAIN_ORG:
            return CORPORATE_DOMAIN_ORG[suffix]
    return None


def build_employer_table(
    persons: pd.DataFrame,
    *,
    org_indexes,
    wikidata_employers: Optional[pd.DataFrame] = None,
    founders: Optional[pd.DataFrame] = None,
    orcid_recent: Optional[pd.DataFrame] = None,
    s2_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Resolve one employer per person via the confidence cascade.

    ``persons`` needs columns ``author_dblp_pid``, ``orcid``, ``homepage_url``.
    ``org_indexes`` is ``(ror_map, alias_map)`` from ``build_indexes``.
    Each optional frame supplies one cascade rung's candidates.
    """
    ror_map, alias_map = org_indexes

    # Precompute per-pid candidates for each rung as {pid: (org_id, as_of_year)}.
    wiki = _name_candidates(wikidata_employers, "author_dblp_pid", "org_label", alias_map)
    found = _founder_candidates(founders)
    orc = _orcid_candidates(orcid_recent, alias_map)
    s2 = _name_candidates(s2_metrics, "author_dblp_pid", "affiliations", alias_map, split="|")

    candidates = {"wikidata": wiki, "founder": found, "orcid": orc, "s2": s2}

    rows: List[Dict[str, Any]] = []
    for person in persons.itertuples():
        pid = getattr(person, "author_dblp_pid")
        chosen = None
        for rung in _ORDER:
            if rung == "homepage":
                org_id = employer_from_homepage(getattr(person, "homepage_url", None), alias_map)
                if org_id:
                    chosen = (org_id, rung, CONFIDENCE[rung], None)
                    break
                continue
            cand = candidates[rung].get(pid)
            if cand and cand[0]:
                chosen = (cand[0], rung, CONFIDENCE[rung], cand[1])
                break
        if chosen:
            rows.append(
                {
                    "author_dblp_pid": pid,
                    "employer_org_id": chosen[0],
                    "source": chosen[1],
                    "confidence": chosen[2],
                    "as_of_year": chosen[3],
                }
            )
    frame = pd.DataFrame(rows, columns=EMPLOYER_COLUMNS)
    LOG.info(
        "stage=employer resolved=%d/%d",
        len(frame),
        len(persons),
    )
    return frame


def _name_candidates(frame, pid_col, name_col, alias_map, split=None):
    out: Dict[str, tuple] = {}
    if frame is None or frame.empty:
        return out
    for row in frame.itertuples():
        pid = getattr(row, pid_col, None)
        raw = getattr(row, name_col, None)
        if not pid or not raw:
            continue
        names = str(raw).split(split) if split else [raw]
        for name in names:
            org_id = match_affiliation(name, alias_map)
            if org_id and pid not in out:
                out[pid] = (org_id, None)
                break
    return out


def _founder_candidates(founders):
    out: Dict[str, tuple] = {}
    if founders is None or founders.empty:
        return out
    for row in founders.itertuples():
        pid = getattr(row, "founder_dblp_pid", None)
        org_id = getattr(row, "org_id", None)
        if pid and org_id and pid not in out:
            out[pid] = (org_id, None)
    return out


def _orcid_candidates(orcid_recent, alias_map):
    out: Dict[str, tuple] = {}
    if orcid_recent is None or orcid_recent.empty:
        return out
    for row in orcid_recent.itertuples():
        # orcid_recent is keyed by orcid; caller must have joined pid on.
        pid = getattr(row, "author_dblp_pid", None)
        name = getattr(row, "employer_name", None)
        year = getattr(row, "start_year", None)
        if not pid or not name:
            continue
        org_id = match_affiliation(name, alias_map)
        if org_id and pid not in out:
            out[pid] = (org_id, year)
    return out
