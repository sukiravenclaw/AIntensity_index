from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.collect import arxiv, csrankings, dblp, openalex, openreview, semantic_scholar
from src.config import ensure_directories
from src.derive.mobility import derive_mobility_events
from src.normalize.orgs import match_organizations, validate_org_file
from src.qa.disambiguation import build_sample
from src.qa.mobility_validation import validate_mobility
from src.qa.report import write_report
from src.schemas import AUTHORSHIPS_SCHEMA, WORKS_SCHEMA, write_parquet


LOG = logging.getLogger("ai_talent_collection.pipeline")


def run(cfg: Dict[str, Any]) -> None:
    ensure_directories(cfg)
    # Fail before any network work if the required curated input is absent.
    validate_org_file(cfg["paths"]["orgs"])
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])

    dblp_records = dblp.collect(cfg)
    _write_interim(dblp_records, interim / "dblp_venue_works.parquet")
    _log_frame("dblp", dblp_records)

    openreview_records = openreview.collect(cfg)
    _write_interim(
        openreview_records,
        interim / "openreview_submissions.parquet",
    )
    _log_frame("openreview", openreview_records)

    arxiv_records = arxiv.collect(cfg)
    _write_interim(arxiv_records, interim / "arxiv_preprints.parquet")
    _log_frame("arxiv", arxiv_records)

    csrankings_records = csrankings.collect(cfg)
    _write_interim(csrankings_records, interim / "csrankings.parquet")
    _log_frame("csrankings", csrankings_records)

    works, authorships = openalex.collect(
        cfg,
        dblp_records,
        openreview_records,
        arxiv_records,
    )
    _log_frame("openalex_works", works)
    _log_frame("openalex_authorships", authorships)

    works, authorships, s2_records = semantic_scholar.enrich(
        cfg,
        works,
        authorships,
    )
    # Citation-gap enrichment must precede collection-local metric fallbacks.
    works = openalex.apply_metric_fallbacks(works)
    if not s2_records.empty:
        serializable = s2_records.copy()
        serializable["external_ids"] = serializable["external_ids"].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        _write_interim(
            serializable,
            interim / "semantic_scholar_crosswalk.parquet",
        )
    _log_frame("semantic_scholar", s2_records)

    authorships, unmatched = match_organizations(
        authorships,
        cfg["paths"]["orgs"],
    )
    disambiguation = build_sample(
        works,
        authorships,
        n_authors=int(cfg["qa"].get("disambiguation_authors", 100)),
        minimum_works=int(cfg["qa"].get("minimum_works", 3)),
        random_seed=int(cfg["project"].get("random_seed", 20220701)),
    )

    # Derive mobility events from authorships
    mobility_events = derive_mobility_events(works, authorships)
    _log_frame("mobility_events", mobility_events)

    # Validate mobility against CSRankings departures
    mobility_validation = validate_mobility(
        mobility_events,
        csrankings_records,
        authorships
    )

    _validate_outputs(works, authorships)
    write_parquet(works, processed / "works.parquet", WORKS_SCHEMA)
    write_parquet(authorships, processed / "authorships.parquet", AUTHORSHIPS_SCHEMA)

    # Write mobility events as parquet
    if not mobility_events.empty:
        mobility_events.to_parquet(
            processed / "mobility_events.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd"
        )

    unmatched.to_csv(processed / "orgs_unmatched.csv", index=False)
    disambiguation.to_csv(processed / "disambiguation_sample.csv", index=False)
    mobility_validation.to_csv(processed / "mobility_validation.csv", index=False)

    write_report(
        cfg["paths"]["report"],
        works,
        authorships,
        unmatched,
        disambiguation,
        mobility_events,
        mobility_validation,
    )
    LOG.info(
        "stage=complete works=%d authorships=%d mobility=%d unmatched=%d qa_rows=%d",
        len(works),
        len(authorships),
        len(mobility_events),
        len(unmatched),
        len(disambiguation),
    )


def _write_interim(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        # Pandas still writes a readable empty Parquet file when columns exist.
        frame.to_parquet(path, index=False, engine="pyarrow")
    else:
        frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _log_frame(stage: str, frame: pd.DataFrame) -> None:
    details = {"stage": stage, "rows": len(frame)}
    for column in ("venue_normalized", "publication_year", "source"):
        if column in frame:
            details[f"distinct_{column}"] = int(frame[column].nunique(dropna=True))
    LOG.info("count=%s", json.dumps(details, sort_keys=True))


def _validate_outputs(works: pd.DataFrame, authorships: pd.DataFrame) -> None:
    if works.empty:
        LOG.warning("validation=empty_works")
        return
    if works["work_id"].isna().any():
        raise ValueError("works.work_id contains nulls")
    if works["work_id"].duplicated().any():
        raise ValueError("works.work_id is not unique")
    unknown = set(authorships["work_id"].dropna()) - set(works["work_id"])
    if unknown:
        raise ValueError(f"authorships contains {len(unknown)} unknown work_id values")
    allowed_decisions = {
        "oral",
        "spotlight",
        "poster",
        "rejected",
        "preprint",
        "unknown",
    }
    invalid = set(works["decision"].dropna()) - allowed_decisions
    if invalid:
        raise ValueError(f"Invalid decision values: {sorted(invalid)}")
