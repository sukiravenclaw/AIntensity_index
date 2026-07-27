from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WORKS_SCHEMA = pa.schema(
    [
        ("work_id", pa.string()),
        ("doi", pa.string()),
        ("arxiv_id", pa.string()),
        ("openreview_id", pa.string()),
        ("s2_paper_id", pa.string()),
        ("title", pa.string()),
        ("publication_year", pa.int32()),
        ("publication_date", pa.date32()),
        ("venue_raw", pa.string()),
        ("venue_normalized", pa.string()),
        ("venue_tier", pa.string()),
        ("decision", pa.string()),
        ("cited_by_count", pa.int64()),
        ("fwci", pa.float64()),
        ("citation_normalized_percentile", pa.float64()),
        ("topics", pa.list_(pa.string())),
        ("primary_topic", pa.string()),
        ("n_authors", pa.int32()),
        ("retrieved_at", pa.timestamp("us", tz="UTC")),
        ("source", pa.string()),
    ]
)

AUTHORSHIPS_SCHEMA = pa.schema(
    [
        ("work_id", pa.string()),
        ("author_openalex_id", pa.string()),
        ("author_s2_id", pa.string()),
        ("orcid", pa.string()),
        ("author_display_name", pa.string()),
        ("author_position", pa.string()),
        ("n_authors", pa.int32()),
        ("n_institutions", pa.int32()),
        ("fractional_credit", pa.float64()),
        ("raw_affiliation_string", pa.string()),
        ("institution_ror", pa.string()),
        ("institution_display_name", pa.string()),
        ("institution_type", pa.string()),
        ("institution_country", pa.string()),
        ("org_id", pa.string()),
        ("retrieved_at", pa.timestamp("us", tz="UTC")),
        ("source", pa.string()),
    ]
)


def empty_frame(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in schema})


def write_parquet(frame: pd.DataFrame, path: str | Path, schema: pa.Schema) -> None:
    output = frame.copy()
    for field in schema:
        if field.name not in output:
            output[field.name] = None
    output = output[[field.name for field in schema]]
    for column in ("retrieved_at",):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    if "publication_date" in output:
        output["publication_date"] = pd.to_datetime(
            output["publication_date"], errors="coerce"
        ).dt.date
    for column in ("publication_year", "n_authors", "n_institutions", "cited_by_count"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ("fwci", "citation_normalized_percentile", "fractional_credit"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    table = pa.Table.from_pandas(output, schema=schema, preserve_index=False, safe=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
