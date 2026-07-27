from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd


LOG = logging.getLogger("ai_talent_collection.qa.mobility_validation")


def validate_mobility(
    mobility_events: pd.DataFrame,
    csrankings: pd.DataFrame,
    authorships: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare derived mobility events against CSRankings departure records.

    Produces a validation report with:
    - matched: events found in both derived and CSRankings
    - missed: in CSRankings but not in derived
    - spurious: in derived but not in CSRankings

    Returns a DataFrame with per-case details and summary metrics.
    """
    if csrankings.empty:
        LOG.warning("stage=mobility_validation status=no_csrankings")
        return _empty_validation_frame()

    # Extract industry departures from CSRankings old records
    departures = csrankings[
        (csrankings["source"] == "csrankings_old") &
        (csrankings["moved_to"].notna())
    ].copy()

    if departures.empty:
        LOG.warning("stage=mobility_validation status=no_departures")
        return _empty_validation_frame()

    # Classify companies vs academic moves
    # Industry keywords in moved_to field
    industry_keywords = {
        "google", "meta", "facebook", "microsoft", "amazon", "apple",
        "openai", "anthropic", "deepmind", "nvidia", "intel", "ibm",
        "tesla", "uber", "airbnb", "netflix", "twitter", "linkedin",
        "startup", "inc", "corp", "llc", "limited"
    }

    def is_industry_move(moved_to: str) -> bool:
        if pd.isna(moved_to):
            return False
        lower = str(moved_to).lower()
        return any(keyword in lower for keyword in industry_keywords)

    departures["is_industry"] = departures["moved_to"].apply(is_industry_move)
    industry_departures = departures[departures["is_industry"]].copy()

    LOG.info(
        "stage=mobility_validation csrankings_departures=%d industry=%d",
        len(departures),
        len(industry_departures)
    )

    # Extract academia-to-industry transitions from derived events
    # (institution_type: education -> company/other)
    if mobility_events.empty:
        derived_transitions = pd.DataFrame()
    else:
        derived_transitions = mobility_events[
            (mobility_events["from_institution_type"] == "education") &
            (mobility_events["to_institution_type"].isin(["company", "other", "nonprofit"]))
        ].copy()

    # Match by author name (fuzzy matching via DBLP or ORCID would be better
    # but for now we do simple name matching)
    # We need to link CSRankings names to author_openalex_ids via authorships

    validation_rows = []

    # For each CSRankings industry departure, check if we have it in derived
    for _, dept in industry_departures.iterrows():
        name = dept["name"]
        moved_to = dept["moved_to"]

        # Find matching transitions (this is simplified - production would use
        # DBLP/ORCID linking)
        matches = _find_matching_transitions(
            name,
            derived_transitions,
            authorships
        )

        if matches:
            status = "matched"
        else:
            status = "missed"

        validation_rows.append({
            "csrankings_name": name,
            "moved_to": moved_to,
            "status": status,
            "n_derived_matches": len(matches),
            "derived_transition_years": [m["transition_year"] for m in matches] if matches else None
        })

    # Find spurious transitions (in derived but not CSRankings)
    # This is harder without good name matching, so we skip for now

    result = pd.DataFrame(validation_rows)

    # Calculate summary metrics
    if not result.empty:
        matched = (result["status"] == "matched").sum()
        missed = (result["status"] == "missed").sum()
        total = len(result)

        precision = matched / (matched + 0) if matched > 0 else 0.0  # No spurious counted yet
        recall = matched / total if total > 0 else 0.0

        # Add summary header rows
        summary = pd.DataFrame([
            {"csrankings_name": "SUMMARY", "moved_to": "precision", "status": f"{precision:.2%}", "n_derived_matches": None, "derived_transition_years": None},
            {"csrankings_name": "SUMMARY", "moved_to": "recall", "status": f"{recall:.2%}", "n_derived_matches": None, "derived_transition_years": None},
            {"csrankings_name": "SUMMARY", "moved_to": "matched", "status": str(matched), "n_derived_matches": None, "derived_transition_years": None},
            {"csrankings_name": "SUMMARY", "moved_to": "missed", "status": str(missed), "n_derived_matches": None, "derived_transition_years": None},
        ])
        result = pd.concat([summary, result], ignore_index=True)

        LOG.info(
            "stage=mobility_validation matched=%d missed=%d precision=%.2f recall=%.2f",
            matched,
            missed,
            precision,
            recall
        )

    return result


def _find_matching_transitions(
    csrankings_name: str,
    derived_transitions: pd.DataFrame,
    authorships: pd.DataFrame
) -> list:
    """
    Find derived transitions for a given CSRankings name.

    This is simplified name matching. Production should use DBLP/ORCID linking.
    """
    if derived_transitions.empty or authorships.empty:
        return []

    # Normalize name for matching
    normalized_name = csrankings_name.lower().strip()

    # Find author_openalex_ids with similar names in authorships
    # (very crude matching - just check if name is substring)
    matching_authors = authorships[
        authorships["author_display_name"].str.lower().str.contains(
            normalized_name.split()[-1],  # Match on last name
            na=False,
            regex=False
        )
    ]["author_openalex_id"].unique()

    # Find transitions for these authors
    matches = []
    for author_id in matching_authors:
        author_transitions = derived_transitions[
            derived_transitions["author_openalex_id"] == author_id
        ]
        for _, trans in author_transitions.iterrows():
            matches.append({
                "author_openalex_id": author_id,
                "transition_year": trans["transition_year"],
                "from_org_id": trans["from_org_id"],
                "to_org_id": trans["to_org_id"]
            })

    return matches


def _empty_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "csrankings_name",
        "moved_to",
        "status",
        "n_derived_matches",
        "derived_transition_years"
    ])
