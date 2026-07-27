from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(
    path: str | Path,
    works: pd.DataFrame,
    authorships: pd.DataFrame,
    unmatched: pd.DataFrame,
    disambiguation: pd.DataFrame,
    mobility_events: pd.DataFrame = None,
    mobility_validation: pd.DataFrame = None,
) -> None:
    if mobility_events is None:
        mobility_events = pd.DataFrame()
    if mobility_validation is None:
        mobility_validation = pd.DataFrame()
    mapped = int(authorships["org_id"].notna().sum()) if not authorships.empty else 0
    mapping_percent = 100 * mapped / len(authorships) if len(authorships) else 0
    sampled_authors = (
        disambiguation["author_openalex_id"].nunique() if not disambiguation.empty else 0
    )
    lines = [
        "# Collection report",
        "",
        "This report describes collection outputs only. It contains no scoring or modeling.",
        "",
        "## Row counts",
        "",
        "| Table | Rows |",
        "|---|---:|",
        f"| `works.parquet` | {len(works):,} |",
        f"| `authorships.parquet` | {len(authorships):,} |",
        f"| `mobility_events.parquet` | {len(mobility_events):,} |",
        f"| `orgs_unmatched.csv` | {len(unmatched):,} |",
        f"| `disambiguation_sample.csv` | {len(disambiguation):,} ({sampled_authors:,} authors) |",
        f"| `mobility_validation.csv` | {len(mobility_validation):,} |",
        "",
        "## Organization mapping",
        "",
        f"{mapped:,} of {len(authorships):,} authorship-institution rows mapped to an "
        f"`org_id` ({mapping_percent:.2f}%).",
        "",
        "## Coverage by venue and year",
        "",
    ]
    if works.empty:
        lines.append("No resolved works.")
    else:
        coverage = (
            works.groupby(["venue_normalized", "publication_year"], dropna=False)
            .size()
            .rename("works")
            .reset_index()
        )
        lines.extend(_markdown_table(coverage))
    lines.extend(["", "## Mobility validation", ""])
    if not mobility_validation.empty:
        # Extract summary metrics from the header rows
        summary_rows = mobility_validation[
            mobility_validation["csrankings_name"] == "SUMMARY"
        ]
        if not summary_rows.empty:
            for _, row in summary_rows.iterrows():
                metric = row["moved_to"]
                value = row["status"]
                lines.append(f"- **{metric.capitalize()}**: {value}")
            lines.append("")
            lines.append(
                "Validation compares derived academia-to-industry transitions against "
                "CSRankings departure records. See `mobility_validation.csv` for details."
            )
        else:
            lines.append("No mobility validation summary available.")
    else:
        lines.append("Mobility validation skipped (CSRankings not available).")

    lines.extend(["", "## Top 50 unmatched affiliation strings", ""])
    if unmatched.empty:
        lines.append("No unmatched non-empty affiliation strings.")
    else:
        lines.extend(
            _markdown_table(
                unmatched[["raw_affiliation_string", "frequency"]].head(50)
            )
        )
    lines.extend(
        [
            "",
            "## Collection notes",
            "",
            "- The publication window and venue tiers are configuration values in `config.yaml`.",
            "- OpenAlex authorships are the only persisted person-linked publication records; no people/profile table is produced.",
            "- Semantic Scholar is used only for paper/author cross-reference IDs and missing citation counts.",
            "- arXiv affiliation fields are not parsed or used.",
            "- OpenAlex currently documents both `fwci` and `citation_normalized_percentile`. If either field disappears from all returned work objects, the pipeline computes a collection-local year-and-primary-topic fallback and records that in `source`.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines

