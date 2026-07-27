from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd


LOG = logging.getLogger("ai_talent_collection.derive.mobility")


def derive_mobility_events(
    works: pd.DataFrame,
    authorships: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct mobility events from publication-listed affiliations.

    Strategy:
    1. For each author, sort their works chronologically
    2. Collapse consecutive identical institutions into a single spell
    3. Emit a transition row for each institution change
    4. Score confidence based on publication density around the transition

    Returns a DataFrame with columns:
    - author_openalex_id
    - from_org_id
    - to_org_id
    - from_institution_type
    - to_institution_type
    - transition_year
    - evidence_work_ids (list)
    - n_works_before
    - n_works_after
    - confidence (low/medium/high)
    """
    if authorships.empty:
        return pd.DataFrame(columns=[
            "author_openalex_id",
            "from_org_id",
            "to_org_id",
            "from_institution_type",
            "to_institution_type",
            "transition_year",
            "evidence_work_ids",
            "n_works_before",
            "n_works_after",
            "confidence"
        ])

    # Join authorships with works to get publication dates
    enriched = authorships.merge(
        works[["work_id", "publication_year", "publication_date"]],
        on="work_id",
        how="left"
    )

    # Filter to rows with valid author_openalex_id and org_id
    enriched = enriched[
        enriched["author_openalex_id"].notna() &
        enriched["org_id"].notna() &
        enriched["publication_year"].notna()
    ].copy()

    if enriched.empty:
        LOG.warning("stage=mobility status=no_valid_authorships")
        return pd.DataFrame(columns=[
            "author_openalex_id",
            "from_org_id",
            "to_org_id",
            "from_institution_type",
            "to_institution_type",
            "transition_year",
            "evidence_work_ids",
            "n_works_before",
            "n_works_after",
            "confidence"
        ])

    # Sort by author and publication year
    enriched = enriched.sort_values(["author_openalex_id", "publication_year", "work_id"])

    transitions = []

    # Group by author
    for author_id, group in enriched.groupby("author_openalex_id", sort=False):
        # Build spells (consecutive periods at same org)
        spells = _build_affiliation_spells(group)

        # Generate transitions between spells
        for i in range(len(spells) - 1):
            from_spell = spells[i]
            to_spell = spells[i + 1]

            # Only emit if org_id changed
            if from_spell["org_id"] != to_spell["org_id"]:
                transition = {
                    "author_openalex_id": author_id,
                    "from_org_id": from_spell["org_id"],
                    "to_org_id": to_spell["org_id"],
                    "from_institution_type": from_spell["institution_type"],
                    "to_institution_type": to_spell["institution_type"],
                    "transition_year": to_spell["first_year"],
                    "evidence_work_ids": from_spell["work_ids"][-2:] + to_spell["work_ids"][:2],
                    "n_works_before": from_spell["n_works"],
                    "n_works_after": to_spell["n_works"],
                    "confidence": _score_confidence(from_spell, to_spell)
                }
                transitions.append(transition)

    result = pd.DataFrame(transitions)

    LOG.info(
        "stage=mobility transitions=%d authors=%d",
        len(result),
        result["author_openalex_id"].nunique() if not result.empty else 0
    )

    return result


def _build_affiliation_spells(author_works: pd.DataFrame) -> List[dict]:
    """
    Collapse consecutive works at the same org into affiliation spells.

    Returns a list of spell dictionaries with:
    - org_id
    - institution_type
    - first_year
    - last_year
    - work_ids
    - n_works
    """
    spells = []
    current_spell = None

    for _, row in author_works.iterrows():
        org_id = row["org_id"]
        year = row["publication_year"]
        work_id = row["work_id"]
        inst_type = row.get("institution_type")

        if current_spell is None or current_spell["org_id"] != org_id:
            # Start new spell
            if current_spell is not None:
                spells.append(current_spell)

            current_spell = {
                "org_id": org_id,
                "institution_type": inst_type,
                "first_year": year,
                "last_year": year,
                "work_ids": [work_id],
                "n_works": 1,
            }
        else:
            # Continue current spell
            current_spell["last_year"] = year
            current_spell["work_ids"].append(work_id)
            current_spell["n_works"] += 1

    # Don't forget the last spell
    if current_spell is not None:
        spells.append(current_spell)

    return spells


def _score_confidence(from_spell: dict, to_spell: dict) -> str:
    """
    Score transition confidence based on publication density.

    Rules:
    - high: 3+ works before AND 3+ works after
    - medium: 2+ works before AND 2+ works after
    - low: otherwise
    """
    before = from_spell["n_works"]
    after = to_spell["n_works"]

    if before >= 3 and after >= 3:
        return "high"
    elif before >= 2 and after >= 2:
        return "medium"
    else:
        return "low"
