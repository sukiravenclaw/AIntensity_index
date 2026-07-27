from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from src.normalize.text import normalize_text


def build_sample(
    works: pd.DataFrame,
    authorships: pd.DataFrame,
    *,
    n_authors: int,
    minimum_works: int,
    random_seed: int,
) -> pd.DataFrame:
    columns = [
        "sample_author_index",
        "author_openalex_id",
        "author_display_name",
        "n_works",
        "work_id",
        "work_title",
        "publication_year",
        "venue_normalized",
        "listed_affiliations",
        "name_collision_candidates",
        "retrieved_at",
        "source",
    ]
    if authorships.empty:
        return pd.DataFrame(columns=columns)
    unique_pairs = authorships[
        ["author_openalex_id", "work_id", "author_display_name"]
    ].drop_duplicates()
    counts = (
        unique_pairs.groupby("author_openalex_id")["work_id"]
        .nunique()
        .loc[lambda series: series >= minimum_works]
    )
    if counts.empty:
        return pd.DataFrame(columns=columns)
    sampled = counts.sample(
        n=min(n_authors, len(counts)),
        random_state=random_seed,
    )

    names = (
        authorships[["author_openalex_id", "author_display_name"]]
        .drop_duplicates("author_openalex_id")
        .copy()
    )
    names["normalized_name"] = names["author_display_name"].map(normalize_text)
    collisions = (
        names.groupby("normalized_name")
        .apply(
            lambda group: [
                {
                    "author_openalex_id": row.author_openalex_id,
                    "author_display_name": row.author_display_name,
                }
                for row in group.itertuples()
            ],
            include_groups=False,
        )
        .to_dict()
    )
    work_lookup = works.set_index("work_id").to_dict("index")
    rows: List[Dict[str, Any]] = []
    for sample_index, (author_id, n_works) in enumerate(sampled.items(), start=1):
        author_rows = authorships[authorships["author_openalex_id"] == author_id]
        display_name = author_rows["author_display_name"].dropna().iloc[0]
        candidates = [
            item
            for item in collisions.get(normalize_text(display_name), [])
            if item["author_openalex_id"] != author_id
        ]
        for work_id, group in author_rows.groupby("work_id"):
            work = work_lookup.get(work_id, {})
            affiliations = sorted(
                {
                    value
                    for value in group["raw_affiliation_string"].dropna().astype(str)
                    if value.strip()
                }
            )
            rows.append(
                {
                    "sample_author_index": sample_index,
                    "author_openalex_id": author_id,
                    "author_display_name": display_name,
                    "n_works": int(n_works),
                    "work_id": work_id,
                    "work_title": work.get("title"),
                    "publication_year": work.get("publication_year"),
                    "venue_normalized": work.get("venue_normalized"),
                    "listed_affiliations": json.dumps(
                        affiliations, ensure_ascii=False, sort_keys=True
                    ),
                    "name_collision_candidates": json.dumps(
                        candidates, ensure_ascii=False, sort_keys=True
                    ),
                    "retrieved_at": group["retrieved_at"].max(),
                    "source": "openalex_authorships;deterministic_qa_sample",
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["sample_author_index", "publication_year", "work_id"]
    )

