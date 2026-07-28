from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.collect import (
    csrankings,
    dblp,
    dblp_dump,
    dblp_person,
    orcid,
    semantic_scholar,
    wikidata,
)
from src.config import ensure_directories
from src.derive import intensity, talent_flow
from src.normalize.org_taxonomy import INDUSTRY_TYPES, SHELL_TYPES
from src.normalize.orgs import validate_org_file
from src.schemas import AUTHORSHIPS_SCHEMA, WORKS_SCHEMA, write_parquet


LOG = logging.getLogger("ai_talent_collection.pipeline")


def run(cfg: Dict[str, Any]) -> None:
    ensure_directories(cfg)
    validate_org_file(cfg["paths"]["orgs"])
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])

    # 1. DBLP: venue-year seed + authorship (co-authorship) graph.
    dblp_works, dblp_auth = dblp.collect_authorships(cfg)
    works = dblp_works.rename(columns={"dblp_work_id": "work_id"})
    authorships = dblp_auth.rename(columns={"dblp_work_id": "work_id"})
    authorships = _finalize_authorships(authorships)
    works = _attach_work_stats(works, authorships)
    _log_frame("dblp_works", works)
    _log_frame("dblp_authorships", authorships)

    # 2. Semantic Scholar: paper citations + author-level metrics keyed to PIDs.
    works, s2_author_metrics, paper_crosswalk = semantic_scholar.collect_citations(
        cfg, works, authorships
    )
    _log_frame("s2_author_metrics", s2_author_metrics)

    # 3. DBLP per-person full history (the training backtrack). Order authors by
    # seed-paper count (desc) so a max_pids cap keeps the highest-signal talent.
    pids = _prioritized_pids(authorships)
    history = _collect_history(cfg, pids, authorships)
    _log_frame("dblp_person_history", history)

    # 4. CSRankings faculty (academic anchor / advisor ground truth).
    faculty = csrankings.collect(cfg)
    _log_frame("csrankings", faculty)

    # 5. ORCID current-employer signal (dated). Scoped to the talent (seed
    # authors) in priority order and capped — ORCID is a per-person crawl, so
    # querying every co-author's ORCID would be an hours-long unscoped sweep.
    orcids = _talent_orcids(cfg, history, pids)
    orcid_emp = orcid.collect(cfg, orcids)
    orcid_recent = orcid.most_recent_employer(orcid_emp)

    # 6. Wikidata founders (required for the founder column) + optional employer facts.
    label_map = _company_labels(cfg["paths"]["orgs"])
    founders = wikidata.collect_founders(cfg, label_map)
    _log_frame("wikidata_founders", founders)
    wikidata_employers = None
    if cfg.get("wikidata", {}).get("person_facts", False):
        facts = wikidata.collect_person_facts(cfg, pids)
        if not facts.empty:
            wikidata_employers = facts[facts["fact_type"] == "employer"]

    # 7. Talent-flow derivation (persons, training, employer, flow matrix).
    tf = talent_flow.run(
        cfg,
        seed_authorships=authorships,
        history=history,
        faculty=faculty,
        s2_metrics=s2_author_metrics,
        wikidata_employers=wikidata_employers,
        founders=founders,
        orcid_recent=orcid_recent,
    )

    # 8. Intensity index.
    intensity.run(
        cfg,
        persons=tf["persons"],
        seed_authorships=authorships,
        works=works,
        employers=tf["employers"],
        training=tf["training"],
    )

    # 9. Persist canonical + interim tables.
    _validate_outputs(works, authorships)
    write_parquet(works, processed / "works.parquet", WORKS_SCHEMA)
    authorships_out = authorships.rename(columns={"author_name": "author_display_name"})
    write_parquet(authorships_out, processed / "authorships.parquet", AUTHORSHIPS_SCHEMA)
    _write_interim(faculty, interim / "csrankings.parquet")
    _write_interim(history, interim / "dblp_person_history.parquet")
    _write_interim(s2_author_metrics, interim / "s2_author_metrics.parquet")
    _write_interim(paper_crosswalk, interim / "s2_paper_crosswalk.parquet")

    _write_report(cfg, works, authorships, tf)
    LOG.info(
        "stage=complete works=%d authorships=%d persons=%d %s",
        len(works),
        len(authorships),
        len(tf["persons"]),
        tf["metrics"],
    )


def _prioritized_pids(authorships: pd.DataFrame) -> list:
    """Unique author PIDs ordered by seed-paper count (desc) — high signal first."""
    if authorships.empty:
        return []
    counts = (
        authorships.dropna(subset=["author_dblp_pid"])
        .groupby("author_dblp_pid")["work_id"].nunique()
        .sort_values(ascending=False)
    )
    return counts.index.tolist()


def _talent_orcids(cfg: Dict[str, Any], history: pd.DataFrame, pids: list) -> list:
    """Valid ORCIDs of seed authors, in priority order, capped by orcid.max_lookups."""
    if history.empty or "author_orcid" not in history.columns:
        return []
    own = history.dropna(subset=["author_dblp_pid", "author_orcid"])
    pid_orcid = (
        own.drop_duplicates("author_dblp_pid")
        .set_index("author_dblp_pid")["author_orcid"].to_dict()
    )
    seen: set = set()
    ordered: list = []
    for pid in pids:  # already highest-signal-first
        cleaned = orcid._clean_orcid(pid_orcid.get(pid))
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    cap = cfg.get("orcid", {}).get("max_lookups")
    if cap:
        ordered = ordered[: int(cap)]
    LOG.info("stage=orcid_scope talent_orcids=%d", len(ordered))
    return ordered


def _collect_history(cfg: Dict[str, Any], pids: list, authorships: pd.DataFrame) -> pd.DataFrame:
    """Full per-person history via the bulk DBLP dump or the per-PID API."""
    source = cfg.get("dblp_person", {}).get("source", "api")
    if source == "dump":
        pid_to_name = (
            authorships.dropna(subset=["author_dblp_pid", "author_name"])
            .drop_duplicates("author_dblp_pid")
            .set_index("author_dblp_pid")["author_name"].to_dict()
        )
        max_pids = cfg.get("dblp_person", {}).get("max_pids")
        target = pids[: int(max_pids)] if max_pids else pids
        return dblp_dump.collect(cfg, target, pid_to_name)
    return dblp_person.collect(cfg, pids)


def _finalize_authorships(authorships: pd.DataFrame) -> pd.DataFrame:
    if authorships.empty:
        authorships["n_authors"] = pd.Series(dtype="int64")
        authorships["fractional_credit"] = pd.Series(dtype="float64")
        return authorships
    n = authorships.groupby("work_id")["author_dblp_pid"].transform("nunique").clip(lower=1)
    authorships = authorships.copy()
    authorships["n_authors"] = n
    authorships["fractional_credit"] = 1.0 / n
    return authorships


def _attach_work_stats(works: pd.DataFrame, authorships: pd.DataFrame) -> pd.DataFrame:
    if works.empty:
        return works
    works = works.copy()
    if "cited_by_count" not in works.columns:
        works["cited_by_count"] = pd.NA  # populated by the S2 stage
    if not authorships.empty:
        n_authors = authorships.groupby("work_id")["author_dblp_pid"].nunique()
        works["n_authors"] = works["work_id"].map(n_authors)
    return works


def _company_labels(org_path: str | Path) -> Dict[str, str]:
    """label -> org_id for non-shell industry orgs (for the Wikidata founder query)."""
    label_map: Dict[str, str] = {}
    with Path(org_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            org_type = (row.get("org_type") or "").strip()
            if org_type not in INDUSTRY_TYPES or org_type in SHELL_TYPES:
                continue
            org_id = (row.get("org_id") or "").strip()
            if not org_id:
                continue
            labels = [row.get("canonical_name") or ""]
            labels += (row.get("aliases") or "").split("|")
            for label in labels:
                label = label.strip()
                if label:
                    label_map.setdefault(label, org_id)
    return label_map


def _write_interim(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        pd.DataFrame(frame if frame is not None else {}).to_parquet(path, index=False, engine="pyarrow")
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


def _write_report(cfg: Dict[str, Any], works, authorships, tf) -> None:
    metrics = tf["metrics"]
    coverage = ""
    if not works.empty and "venue_normalized" in works and "publication_year" in works:
        counts = (
            works.groupby(["venue_normalized", "publication_year"]).size()
            .reset_index(name="works").sort_values(["venue_normalized", "publication_year"])
        )
        coverage = "\n".join(
            f"| {r.venue_normalized} | {int(r.publication_year)} | {int(r.works)} |"
            for r in counts.itertuples()
        )
    lines = [
        "# Collection report (OpenAlex-free, person-centric)",
        "",
        "| Table | Rows |",
        "|---|---:|",
        f"| works.parquet | {len(works):,} |",
        f"| authorships.parquet | {len(authorships):,} |",
        f"| flow/persons.parquet | {metrics['total_talent']:,} |",
        "",
        "## Talent coverage",
        "",
        f"- With training lab: {metrics['with_training']:,} ({metrics['pct_with_training']}%)",
        f"- With current employer: {metrics['with_employer']:,} ({metrics['pct_with_employer']}%)",
        f"- Founders: {metrics['founders']:,}",
        "",
        "See `flow/FLOW_REPORT.md` and `flow/INTENSITY_REPORT.md` for detail.",
        "",
        "## Coverage by venue and year",
        "",
        "| venue_normalized | publication_year | works |",
        "|---|---|---|",
        coverage,
    ]
    Path(cfg["paths"]["report"]).write_text("\n".join(lines), encoding="utf-8")
