from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.common import RawCacheClient, utc_now
from src.normalize.orgs import build_indexes, match_affiliation
from src.normalize.text import normalize_text


LOG = logging.getLogger("ai_talent_collection.csrankings")

# CSRankings is CC BY-NC-ND 4.0 - NoDerivatives
# We fetch at build time and use for validation seeding only
# Do not vendor transformed copies

CSRANKINGS_CURRENT_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/csrankings.csv"
CSRANKINGS_OLD_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/old/old-csrankings.csv"


def collect(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch CSRankings data for faculty seeding and mobility validation.

    Returns a DataFrame with columns:
    - name: author name
    - affiliation: current institution
    - homepage: personal page
    - scholarid: Google Scholar ID
    - dblp: DBLP author ID
    - orcid: ORCID
    - moved_to: company name if departed to industry (from old/ records)
    - source: 'csrankings_current' or 'csrankings_old'
    - retrieved_at: timestamp
    """
    if not cfg.get("csrankings", {}).get("enabled", True):
        LOG.info("stage=csrankings status=disabled")
        return pd.DataFrame()

    external_dir = Path(cfg["paths"].get("external", "data/external"))
    external_dir.mkdir(parents=True, exist_ok=True)

    # Fetch current faculty
    current_records = _fetch_csrankings_file(
        cfg,
        CSRANKINGS_CURRENT_URL,
        "csrankings_current",
        external_dir / "csrankings.csv"
    )

    # Fetch old records (departures) - handle 404 gracefully
    try:
        old_records = _fetch_csrankings_file(
            cfg,
            CSRANKINGS_OLD_URL,
            "csrankings_old",
            external_dir / "old-csrankings.csv"
        )
    except Exception as exc:
        LOG.warning("stage=csrankings old_fetch_failed error=%s", exc)
        old_records = pd.DataFrame()

    if old_records.empty:
        combined = current_records
    else:
        combined = pd.concat([current_records, old_records], ignore_index=True)

    combined = _annotate_faculty(combined, cfg)

    LOG.info(
        "stage=csrankings current=%d old=%d total=%d org_mapped=%d",
        len(current_records),
        len(old_records),
        len(combined),
        int(combined["org_id"].notna().sum()) if "org_id" in combined else 0,
    )

    return combined


def _annotate_faculty(combined: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add a DBLP-canonical normalized name and an org_id for each faculty row.

    ``normalized_name`` is the join key to the DBLP co-authorship graph (CSRankings
    names are DBLP-canonical, carrying disambiguation suffixes). ``org_id`` maps
    the faculty's institution to the curated org registry so a detected advisor
    yields a training institution.
    """
    if combined.empty:
        combined["normalized_name"] = pd.Series(dtype="object")
        combined["org_id"] = pd.Series(dtype="object")
        return combined

    combined = combined.copy()
    combined["normalized_name"] = combined["name"].map(normalize_text)

    org_path = cfg["paths"]["orgs"]
    try:
        ror_map, alias_map = build_indexes(org_path)
    except Exception as exc:  # missing/invalid org file should not kill collection
        LOG.warning("stage=csrankings org_index_failed=%s", exc)
        combined["org_id"] = None
        return combined

    unique_affils = combined["affiliation"].dropna().unique()
    affil_to_org = {
        affil: match_affiliation(affil, alias_map, ror_map=ror_map)
        for affil in unique_affils
    }
    combined["org_id"] = combined["affiliation"].map(affil_to_org)
    return combined


def _fetch_csrankings_file(
    cfg: Dict[str, Any],
    url: str,
    source_name: str,
    local_path: Path,
) -> pd.DataFrame:
    """Fetch a single CSRankings CSV file and parse it."""
    user_agent = cfg.get("_user_agent", cfg["project"]["user_agent"])

    client = RawCacheClient(
        source="csrankings",
        raw_root=cfg["paths"]["raw"],
        user_agent=user_agent,
        delay_seconds=cfg["rate_limits"].get("csrankings_seconds", 2.0),
        timeout_seconds=cfg["rate_limits"].get("timeout_seconds", 90),
        max_retries=cfg["rate_limits"].get("max_retries", 6),
    )

    response = client.request("GET", url, extension="csv")
    retrieved_at = response.retrieved_at

    # Write to external directory for reference
    local_path.write_bytes(response.content)
    LOG.info("stage=csrankings source=%s path=%s", source_name, local_path)

    # Parse CSV
    try:
        df = pd.read_csv(local_path, encoding="utf-8")
    except Exception as exc:
        LOG.warning("stage=csrankings source=%s parse_error=%s", source_name, exc)
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Add metadata
    df["source"] = source_name
    df["retrieved_at"] = retrieved_at

    # For old records, extract company from affiliation if it looks like industry
    if source_name == "csrankings_old":
        df["moved_to"] = df.get("affiliation", pd.Series(dtype=str))
    else:
        df["moved_to"] = None

    # Standardize expected columns
    expected = ["name", "affiliation", "homepage", "scholarid", "dblp", "orcid"]
    for col in expected:
        if col not in df:
            df[col] = None

    return df[["name", "affiliation", "homepage", "scholarid", "dblp", "orcid", "moved_to", "source", "retrieved_at"]]
