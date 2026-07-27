from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.normalize.text import normalize_ror, normalize_text


LOG = logging.getLogger("ai_talent_collection.orgs")


class OrganizationFileError(ValueError):
    pass


def validate_org_file(org_path: str | Path) -> None:
    _load_orgs(org_path)


def match_organizations(
    authorships: pd.DataFrame,
    org_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    orgs = _load_orgs(org_path)
    ror_map: Dict[str, str] = {}
    alias_map: Dict[str, str] = {}
    for org in orgs:
        org_id = org["org_id"]
        ror = normalize_ror(org.get("ror"))
        if ror:
            _insert_unique(ror_map, ror, org_id, "ROR")
        aliases = [org.get("display_name")] + _split_aliases(org.get("aliases"))
        for alias in aliases:
            normalized = normalize_text(alias)
            if normalized:
                _insert_unique(alias_map, normalized, org_id, "alias")

    output = authorships.copy()
    if output.empty:
        output["org_id"] = pd.Series(dtype="object")
        unmatched = pd.DataFrame(
            columns=["raw_affiliation_string", "frequency", "retrieved_at", "source"]
        )
        return output, unmatched

    matched_ids: List[Optional[str]] = []
    methods: List[Optional[str]] = []
    for row in output.itertuples():
        ror = normalize_ror(row.institution_ror)
        org_id = ror_map.get(ror) if ror else None
        method = "ror_exact" if org_id else None
        if not org_id:
            for candidate in (
                row.raw_affiliation_string,
                row.institution_display_name,
            ):
                normalized = normalize_text(candidate)
                if normalized and normalized in alias_map:
                    org_id = alias_map[normalized]
                    method = "alias_normalized_exact"
                    break
        matched_ids.append(org_id)
        methods.append(method)
    output["org_id"] = matched_ids
    # Preserve matching provenance in source without adding an unrequested field.
    output["source"] = [
        f"{source};org_match:{method}" if method else source
        for source, method in zip(output["source"], methods)
    ]

    unmatched_rows = output[
        output["org_id"].isna()
        & output["raw_affiliation_string"].notna()
        & (output["raw_affiliation_string"].astype(str).str.strip() != "")
    ]
    if unmatched_rows.empty:
        unmatched = pd.DataFrame(
            columns=["raw_affiliation_string", "frequency", "retrieved_at", "source"]
        )
    else:
        unmatched = (
            unmatched_rows.groupby("raw_affiliation_string", dropna=False)
            .agg(
                frequency=("work_id", "size"),
                retrieved_at=("retrieved_at", "max"),
            )
            .reset_index()
        )
        unmatched["source"] = "openalex_authorships;org_match_unmatched"
        unmatched = unmatched.sort_values(
            ["frequency", "raw_affiliation_string"], ascending=[False, True]
        )
    mapped = int(output["org_id"].notna().sum())
    LOG.info(
        "stage=org_matching rows=%d mapped=%d percent=%.2f",
        len(output),
        mapped,
        100 * mapped / len(output) if len(output) else 0,
    )
    return output, unmatched


def _load_orgs(path: str | Path) -> List[Dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        raise OrganizationFileError(f"Missing hand-curated organization file: {candidate}")
    with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if "org_id" not in columns:
            raise OrganizationFileError("data/orgs.csv must contain an org_id column")
        if not columns.intersection({"ror", "institution_ror", "ror_id", "display_name", "aliases"}):
            raise OrganizationFileError(
                "data/orgs.csv needs at least one matching column: ror, display_name, or aliases"
            )
        rows: List[Dict[str, str]] = []
        for row in reader:
            if not (row.get("org_id") or "").strip():
                continue
            rows.append(
                {
                    "org_id": row["org_id"].strip(),
                    "display_name": (row.get("display_name") or row.get("name") or "").strip(),
                    "ror": (
                        row.get("ror")
                        or row.get("institution_ror")
                        or row.get("ror_id")
                        or ""
                    ).strip(),
                    "aliases": (row.get("aliases") or row.get("alias") or "").strip(),
                }
            )
    if not rows:
        raise OrganizationFileError(
            f"{candidate} has no organization rows. Replace the checked-in header template."
        )
    return rows


def _split_aliases(value: Any) -> List[str]:
    if not value:
        return []
    text = str(value)
    separator = "|" if "|" in text else ";"
    return [piece.strip() for piece in text.split(separator) if piece.strip()]


def _insert_unique(
    mapping: Dict[str, str],
    key: str,
    org_id: str,
    label: str,
) -> None:
    previous = mapping.get(key)
    if previous and previous != org_id:
        raise OrganizationFileError(
            f"Ambiguous {label} {key!r} maps to both {previous!r} and {org_id!r}"
        )
    mapping[key] = org_id
