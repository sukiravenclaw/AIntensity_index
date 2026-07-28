"""Talent-flow derivation (OpenAlex-free, person-centric model).

Institution is a property of the *person*, reconstructed from the DBLP full
publication history + the CSRankings faculty co-authorship graph. Supersedes the
affiliation-timeline prototype ``src/flow/derive_talent_flow.py``.

Pipeline of pure functions (unit-testable with small frames):

1. ``build_persons``    — the talent set (seed authors) with identity/metrics.
2. ``derive_training``  — highest-education lab: the CSRankings-faculty co-author
                          cluster in the person's earliest sustained window.
3. ``derive_employers`` — current employer via the confidence cascade + founder
                          overlay (``src/derive/employer.py``).
4. ``build_flow_matrix``— lab -> employer counts with cell suppression.
5. ``coverage_cascade`` — where the talent is lost, unweighted and citation-weighted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.derive.employer import build_employer_table
from src.normalize.org_taxonomy import OrgTaxonomy
from src.normalize.orgs import build_indexes
from src.normalize.text import normalize_text
from src.schemas import (
    PERSON_EMPLOYER_SCHEMA,
    PERSON_TRAINING_SCHEMA,
    PERSONS_SCHEMA,
    write_parquet,
)

LOG = logging.getLogger("ai_talent_collection.talent_flow")

EARLY_WINDOW_YEARS = 3     # first_pub_year .. first_pub_year + N = "doctoral-era"
MIN_EARLY_COPUBS = 2       # sustained co-authorship required to call someone an advisor
MIN_CELL_SIZE = 3          # suppression threshold for the published matrix


# --------------------------------------------------------------------------- #
# 1. Persons
# --------------------------------------------------------------------------- #
def build_persons(
    seed_authorships: pd.DataFrame,
    history: pd.DataFrame,
    s2_metrics: Optional[pd.DataFrame] = None,
    faculty: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """The talent set: distinct seed authors with identity + career metrics."""
    seed = seed_authorships.dropna(subset=["author_dblp_pid"])
    pids = seed["author_dblp_pid"].unique()

    # Best display name per pid (from seed, else history).
    name_src = pd.concat(
        [seed[["author_dblp_pid", "author_name"]],
         history[["author_dblp_pid", "author_name"]]],
        ignore_index=True,
    ).dropna()
    display = name_src.groupby("author_dblp_pid")["author_name"].first()

    # Career span + volume from full history.
    hist = history[history["author_dblp_pid"].isin(pids)]
    first_year = hist.groupby("author_dblp_pid")["publication_year"].min()
    n_pubs = hist.groupby("author_dblp_pid")["dblp_work_id"].nunique()
    n_seed = seed.groupby("author_dblp_pid")["work_id"].nunique() if "work_id" in seed else \
        seed.groupby("author_dblp_pid")["dblp_work_id"].nunique()

    persons = pd.DataFrame({"author_dblp_pid": pids})
    persons["display_name"] = persons["author_dblp_pid"].map(display)
    persons["first_pub_year"] = persons["author_dblp_pid"].map(first_year)
    persons["n_pubs"] = persons["author_dblp_pid"].map(n_pubs).fillna(0).astype(int)
    persons["n_pubs_seed"] = persons["author_dblp_pid"].map(n_seed).fillna(0).astype(int)

    # Citation metrics from S2 (keyed by pid).
    if s2_metrics is not None and not s2_metrics.empty:
        m = s2_metrics.drop_duplicates("author_dblp_pid").set_index("author_dblp_pid")
        persons["citations_total"] = persons["author_dblp_pid"].map(m["citation_count"])
        persons["h_index"] = persons["author_dblp_pid"].map(m["h_index"])
        persons["s2_author_id"] = persons["author_dblp_pid"].map(m["s2_author_id"])
    else:
        persons["citations_total"] = np.nan
        persons["h_index"] = np.nan
        persons["s2_author_id"] = None

    # ORCID: prefer the DBLP-declared orcid on the person's own signatures.
    persons["orcid"] = None
    persons["homepage_url"] = None
    if "author_orcid" in history.columns:
        own = history[history["author_dblp_pid"].isin(pids)].dropna(subset=["author_orcid"])
        pid_orcid = own.groupby("author_dblp_pid")["author_orcid"].first()
        persons["orcid"] = persons["author_dblp_pid"].map(pid_orcid)

    # Fall back to CSRankings faculty rows that match this person by name.
    if faculty is not None and not faculty.empty:
        fac = faculty.copy()
        fac["normalized_name"] = fac.get("normalized_name", fac["name"].map(normalize_text))
        fac = fac.dropna(subset=["normalized_name"]).drop_duplicates("normalized_name")
        fac = fac.set_index("normalized_name")
        norm = persons["display_name"].map(normalize_text)
        if "orcid" in fac:
            persons["orcid"] = persons["orcid"].combine_first(norm.map(fac["orcid"]))
        if "homepage" in fac:
            persons["homepage_url"] = norm.map(fac["homepage"])
    persons["source"] = "derived"
    return persons


# --------------------------------------------------------------------------- #
# 2. Training / advisor backtrack
# --------------------------------------------------------------------------- #
def build_faculty_map(faculty: pd.DataFrame) -> Dict[str, str]:
    """normalized faculty name -> org_id (first non-null)."""
    if faculty is None or faculty.empty:
        return {}
    fac = faculty.copy()
    if "normalized_name" not in fac:
        fac["normalized_name"] = fac["name"].map(normalize_text)
    fac = fac.dropna(subset=["normalized_name", "org_id"])
    out: Dict[str, str] = {}
    for row in fac.itertuples():
        out.setdefault(row.normalized_name, row.org_id)
    return out


def derive_training(
    persons: pd.DataFrame,
    history: pd.DataFrame,
    faculty_map: Dict[str, str],
) -> pd.DataFrame:
    """Attribute each person's highest-education lab from their earliest window.

    The advisor is the CSRankings faculty co-author with the most co-authored
    works in the person's first ``EARLY_WINDOW_YEARS`` of publishing (last-author
    frequency breaks ties). The training institution is that faculty's org_id.
    """
    rows: List[Dict[str, Any]] = []
    hist = history.dropna(subset=["author_dblp_pid"])
    works_by_person = hist.groupby("author_dblp_pid")
    # work_id -> list of (pid, name, position)
    authors_by_work = hist.groupby("dblp_work_id")

    first_year = persons.set_index("author_dblp_pid")["first_pub_year"].to_dict()

    for pid, start in first_year.items():
        if pd.isna(start) or pid not in works_by_person.groups:
            continue
        person_rows = works_by_person.get_group(pid)
        window_end = start + EARLY_WINDOW_YEARS
        early = person_rows[
            (person_rows["publication_year"] >= start)
            & (person_rows["publication_year"] <= window_end)
        ]
        early_work_ids = set(early["dblp_work_id"])
        if not early_work_ids:
            continue

        # Faculty co-authors on the early works.
        candidates: Dict[str, Dict[str, Any]] = {}
        for work_id in early_work_ids:
            if work_id not in authors_by_work.groups:
                continue
            for co in authors_by_work.get_group(work_id).itertuples():
                if co.author_dblp_pid == pid:
                    continue
                norm = normalize_text(co.author_name)
                org_id = faculty_map.get(norm)
                if not org_id:
                    continue
                c = candidates.setdefault(
                    co.author_dblp_pid,
                    {"name": co.author_name, "org_id": org_id, "copubs": 0, "last": 0},
                )
                c["copubs"] += 1
                if getattr(co, "author_position", None) == "last":
                    c["last"] += 1

        viable = {k: v for k, v in candidates.items() if v["copubs"] >= MIN_EARLY_COPUBS}
        if not viable:
            continue
        ranked = sorted(
            viable.items(),
            key=lambda kv: (kv[1]["copubs"], kv[1]["last"]),
            reverse=True,
        )
        top_pid, top = ranked[0]
        # The bulk-dump path has no co-author PIDs; keep labs distinct by advisor
        # name so a nameless PID doesn't collapse different advisors together.
        if not top_pid:
            top_pid = "name:" + normalize_text(top["name"])
        alt_pid = ranked[1][0] if len(ranked) > 1 else None
        # Confidence: more early co-pubs + last-author signal => higher.
        confidence = min(1.0, 0.4 + 0.15 * top["copubs"] + (0.15 if top["last"] else 0.0))
        rows.append(
            {
                "author_dblp_pid": pid,
                "training_org_id": top["org_id"],
                "advisor_dblp_pid": top_pid,
                "advisor_name": top["name"],
                "advisor_alt_dblp_pid": alt_pid,
                "method": "csrankings_coauthor",
                "confidence": round(confidence, 3),
                "n_advisor_copubs": int(top["copubs"]),
                "training_start_year": int(start),
                "training_end_year": int(min(window_end, person_rows["publication_year"].max())),
            }
        )
    columns = [
        "author_dblp_pid", "training_org_id", "advisor_dblp_pid", "advisor_name",
        "advisor_alt_dblp_pid", "method", "confidence", "n_advisor_copubs",
        "training_start_year", "training_end_year",
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# 3. Employer + founder overlay
# --------------------------------------------------------------------------- #
def derive_employers(
    persons: pd.DataFrame,
    org_indexes,
    *,
    wikidata_employers: Optional[pd.DataFrame] = None,
    founders: Optional[pd.DataFrame] = None,
    orcid_recent: Optional[pd.DataFrame] = None,
    s2_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Employer cascade + founder flag. founders is the Wikidata founder frame."""
    # Join ORCID most-recent employer onto pid via persons.orcid.
    orcid_joined = None
    if orcid_recent is not None and not orcid_recent.empty:
        orcid_joined = orcid_recent.merge(
            persons[["author_dblp_pid", "orcid"]].dropna(subset=["orcid"]),
            on="orcid", how="inner",
        )

    # Founder candidates keyed to pid: prefer explicit founder_dblp_pid, else
    # match founder by ORCID, else by normalized name.
    founder_pid = _founders_to_pid(founders, persons)

    employers = build_employer_table(
        persons,
        org_indexes=org_indexes,
        wikidata_employers=wikidata_employers,
        founders=founder_pid,
        orcid_recent=orcid_joined,
        s2_metrics=s2_metrics,
    )

    # Founder overlay columns.
    founded = {}
    if founder_pid is not None and not founder_pid.empty:
        for row in founder_pid.itertuples():
            founded[getattr(row, "founder_dblp_pid")] = getattr(row, "org_id")
    employers["is_founder"] = employers["author_dblp_pid"].isin(founded).astype(bool)
    employers["founded_org_id"] = employers["author_dblp_pid"].map(founded)
    return employers


def _founders_to_pid(founders: Optional[pd.DataFrame], persons: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resolve founder rows to a DBLP pid using pid, then orcid, then name."""
    if founders is None or founders.empty:
        return founders
    out = founders.copy()
    # name/orcid lookups from persons
    by_orcid = persons.dropna(subset=["orcid"]).set_index("orcid")["author_dblp_pid"].to_dict()
    name_to_pid = {
        normalize_text(n): p
        for p, n in zip(persons["author_dblp_pid"], persons["display_name"])
        if isinstance(n, str)
    }
    pids = []
    for row in out.itertuples():
        pid = getattr(row, "founder_dblp_pid", None)
        if not pid and getattr(row, "founder_orcid", None):
            pid = by_orcid.get(str(row.founder_orcid).split("orcid.org/")[-1].strip("/"))
        if not pid and getattr(row, "founder_name", None):
            pid = name_to_pid.get(normalize_text(row.founder_name))
        pids.append(pid)
    out["founder_dblp_pid"] = pids
    return out.dropna(subset=["founder_dblp_pid"])


# --------------------------------------------------------------------------- #
# 4. Flow matrix
# --------------------------------------------------------------------------- #
def build_flow_matrix(
    training: pd.DataFrame,
    employers: pd.DataFrame,
    taxonomy: OrgTaxonomy,
    min_cell_size: int = MIN_CELL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """lab (training_org + advisor) -> employer counts, with suppression.

    Founders are suppression-exempt (public figures). Returns (full, published).
    """
    if training.empty or employers.empty:
        cols = ["training_org_id", "advisor_dblp_pid", "employer_org_id",
                "employer_type", "is_industry", "n", "lab_total", "share_of_lab",
                "n_founders", "suppressed"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    joined = training.merge(
        employers[["author_dblp_pid", "employer_org_id", "is_founder"]],
        on="author_dblp_pid", how="inner",
    )
    joined = joined[joined["employer_org_id"].notna()]
    joined = joined[joined["employer_org_id"].map(taxonomy.is_valid_destination)]
    if joined.empty:
        cols = ["training_org_id", "advisor_dblp_pid", "employer_org_id",
                "employer_type", "is_industry", "n", "lab_total", "share_of_lab",
                "n_founders", "suppressed"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    grouped = joined.groupby(["training_org_id", "advisor_dblp_pid", "employer_org_id"])
    cells = grouped.agg(
        n=("author_dblp_pid", "nunique"),
        n_founders=("is_founder", "sum"),
    ).reset_index()

    lab_totals = joined.groupby(["training_org_id", "advisor_dblp_pid"])["author_dblp_pid"].nunique()
    cells["lab_total"] = cells.set_index(["training_org_id", "advisor_dblp_pid"]).index.map(lab_totals)
    cells["share_of_lab"] = cells["n"] / cells["lab_total"]
    cells["employer_type"] = cells["employer_org_id"].map(taxonomy.org_type)
    cells["is_industry"] = cells["employer_org_id"].map(taxonomy.is_industry)
    # Suppress small cells unless they contain a founder.
    cells["suppressed"] = (cells["n"] < min_cell_size) & (cells["n_founders"] == 0)

    published = cells[~cells["suppressed"]].copy()
    return cells, published


# --------------------------------------------------------------------------- #
# 5. Coverage cascade
# --------------------------------------------------------------------------- #
def coverage_cascade(
    persons: pd.DataFrame,
    training: pd.DataFrame,
    employers: pd.DataFrame,
) -> Dict[str, Any]:
    total = len(persons)
    weight = persons.set_index("author_dblp_pid")["citations_total"].fillna(0)
    total_weight = float(weight.sum()) or 1.0

    trained = set(training["author_dblp_pid"])
    employed = set(employers.loc[employers["employer_org_id"].notna(), "author_dblp_pid"])
    founders = set(employers.loc[employers.get("is_founder", False) == True, "author_dblp_pid"])  # noqa: E712

    def wshare(pids):
        return round(100 * float(weight[weight.index.isin(pids)].sum()) / total_weight, 1)

    return {
        "total_talent": total,
        "with_training": len(trained),
        "with_employer": len(employed),
        "with_training_and_employer": len(trained & employed),
        "founders": len(founders),
        "pct_with_training": round(100 * len(trained) / max(total, 1), 1),
        "pct_with_employer": round(100 * len(employed) / max(total, 1), 1),
        "wpct_with_training": wshare(trained),
        "wpct_with_employer": wshare(employed),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    cfg: Dict[str, Any],
    *,
    seed_authorships: pd.DataFrame,
    history: pd.DataFrame,
    faculty: pd.DataFrame,
    s2_metrics: Optional[pd.DataFrame] = None,
    wikidata_employers: Optional[pd.DataFrame] = None,
    founders: Optional[pd.DataFrame] = None,
    orcid_recent: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Derive persons/training/employer/flow tables and write them to disk."""
    org_path = cfg["paths"]["orgs"]
    taxonomy = OrgTaxonomy.from_csv(org_path)
    org_indexes = build_indexes(org_path)

    persons = build_persons(seed_authorships, history, s2_metrics, faculty)
    faculty_map = build_faculty_map(faculty)
    training = derive_training(persons, history, faculty_map)
    employers = derive_employers(
        persons, org_indexes,
        wikidata_employers=wikidata_employers,
        founders=founders,
        orcid_recent=orcid_recent,
        s2_metrics=s2_metrics,
    )
    flow_full, flow_published = build_flow_matrix(training, employers, taxonomy)
    metrics = coverage_cascade(persons, training, employers)

    flow_dir = Path(cfg["paths"]["processed"]) / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(persons, flow_dir / "persons.parquet", PERSONS_SCHEMA)
    write_parquet(training, flow_dir / "person_training.parquet", PERSON_TRAINING_SCHEMA)
    write_parquet(employers, flow_dir / "person_employer.parquet", PERSON_EMPLOYER_SCHEMA)
    flow_full.to_parquet(flow_dir / "flow_matrix.parquet", index=False)
    flow_published.to_csv(flow_dir / "flow_matrix_published.csv", index=False)
    _write_flow_report(cfg, metrics, flow_full, flow_published, training, taxonomy)

    LOG.info("stage=talent_flow %s", metrics)
    return {
        "persons": persons,
        "training": training,
        "employers": employers,
        "flow_full": flow_full,
        "flow_published": flow_published,
        "metrics": metrics,
    }


def _write_flow_report(cfg, metrics, flow_full, flow_published, training, taxonomy) -> None:
    n_industry = int(flow_published["is_industry"].sum()) if not flow_published.empty else 0
    lines = [
        "# Talent Flow Report",
        "",
        "Person-centric derivation (OpenAlex-free). Institution is reconstructed",
        "from DBLP full history + the CSRankings faculty co-authorship graph.",
        "",
        "## Coverage cascade",
        "",
        "| Stage | Count | % | Citation-weighted % |",
        "|---|---:|---:|---:|",
        f"| Talent (seed authors) | {metrics['total_talent']:,} | 100.0 | 100.0 |",
        f"| With training lab | {metrics['with_training']:,} | {metrics['pct_with_training']} | {metrics['wpct_with_training']} |",
        f"| With current employer | {metrics['with_employer']:,} | {metrics['pct_with_employer']} | {metrics['wpct_with_employer']} |",
        f"| With both (flow edge) | {metrics['with_training_and_employer']:,} | | |",
        f"| Founders | {metrics['founders']:,} | | |",
        "",
        "## Flow matrix",
        "",
        f"- Total cells: {len(flow_full):,}",
        f"- Published cells (n >= {MIN_CELL_SIZE} or founder): {len(flow_published):,}",
        f"- Published industry-destination cells: {n_industry:,}",
        "",
        "## Method notes",
        "",
        "- Training lab = CSRankings-faculty co-author cluster in the person's first",
        f"  {EARLY_WINDOW_YEARS} publishing years (>= {MIN_EARLY_COPUBS} co-pubs). A doctoral-era",
        "  advisor proxy, not a verified enrollment record.",
        "- Employer resolved by cascade: wikidata > founder > orcid > homepage > s2.",
        "- Founders are exempt from cell suppression (public figures).",
    ]
    report_path = Path(cfg["paths"]["processed"]) / "flow" / "FLOW_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
