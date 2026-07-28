"""Per-company AI talent-intensity index.

Per-person weight (heavy-tailed citations are log-damped, then z-scored *within
venue family* so a CVPR author and an ACL author are comparable):

    cite_p = ln(1 + fractional_citations_p)
    pub_p  = ln(1 + fractional_pubs_p)
    w_p    = ALPHA * z_family(cite_p) + (1 - ALPHA) * z_family(pub_p)   (shifted >= 0)

Per-company intensity, employer-confidence weighted, with a founder bonus:

    I_c = sum_{employer(p)=c} w_p * conf_{p,c}  +  BETA * sum_{founder(p,c)} w_p

Fractional credit (1 / n_authors on each seed paper) prevents 200-author papers
from inflating any single person. Outputs raw, 0-100 index, share, and per-capita
normalizations plus a lab-decomposition of each company's intensity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.normalize.org_taxonomy import OrgTaxonomy

LOG = logging.getLogger("ai_talent_collection.intensity")

ALPHA = 0.7   # weight on citations vs publication count
BETA = 1.0    # founder bonus multiplier

VENUE_FAMILY = {
    "CVPR": "vision", "ICCV": "vision", "ECCV": "vision",
    "ACL": "nlp", "EMNLP": "nlp", "COLM": "nlp",
    "NeurIPS": "ml", "ICML": "ml", "ICLR": "ml", "AAAI": "ml",
}


def _work_id_col(frame: pd.DataFrame) -> str:
    return "work_id" if "work_id" in frame.columns else "dblp_work_id"


def compute_person_weights(
    persons: pd.DataFrame,
    seed_authorships: pd.DataFrame,
    works: pd.DataFrame,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Attach fractional citations/pubs, venue family, and the z-scored weight."""
    wid_w = _work_id_col(works)
    wid_a = _work_id_col(seed_authorships)

    works = works.copy()
    if "cited_by_count" not in works.columns:
        works["cited_by_count"] = 0
    if "venue_normalized" not in works.columns:
        works["venue_normalized"] = None

    # n_authors per seed work (fractional credit denominator).
    n_authors = seed_authorships.groupby(wid_a)["author_dblp_pid"].nunique().rename("n_authors")
    w = works[[wid_w, "venue_normalized", "cited_by_count"]].copy()
    w = w.merge(n_authors, left_on=wid_w, right_index=True, how="left")
    w["n_authors"] = w["n_authors"].fillna(1).clip(lower=1)
    w["cited_by_count"] = pd.to_numeric(w["cited_by_count"], errors="coerce").fillna(0)
    w["cite_credit"] = w["cited_by_count"] / w["n_authors"]
    w["pub_credit"] = 1.0 / w["n_authors"]
    w["family"] = w["venue_normalized"].map(VENUE_FAMILY).fillna("other")

    a = seed_authorships[[wid_a, "author_dblp_pid"]].merge(
        w[[wid_w, "cite_credit", "pub_credit", "family"]],
        left_on=wid_a, right_on=wid_w, how="left",
    )
    agg = a.groupby("author_dblp_pid").agg(
        fractional_citations=("cite_credit", "sum"),
        fractional_pubs=("pub_credit", "sum"),
    )
    # Primary venue family = the family the person publishes in most.
    fam = (
        a.dropna(subset=["family"])
        .groupby(["author_dblp_pid", "family"]).size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("author_dblp_pid")
        .set_index("author_dblp_pid")["family"]
    )

    out = persons.copy()
    out["fractional_citations"] = out["author_dblp_pid"].map(agg["fractional_citations"]).fillna(0.0)
    out["fractional_pubs"] = out["author_dblp_pid"].map(agg["fractional_pubs"]).fillna(0.0)
    out["venue_family"] = out["author_dblp_pid"].map(fam).fillna("other")
    out["cite_log"] = np.log1p(out["fractional_citations"])
    out["pub_log"] = np.log1p(out["fractional_pubs"])
    out["cite_z"] = _z_within(out, "cite_log", "venue_family")
    out["pub_z"] = _z_within(out, "pub_log", "venue_family")
    raw = alpha * out["cite_z"] + (1 - alpha) * out["pub_z"]
    # Shift to non-negative so downstream sums are monotone.
    shift = raw.min()
    out["weight"] = raw - shift if pd.notna(shift) else 0.0
    return out


def _z_within(frame: pd.DataFrame, col: str, group: str) -> pd.Series:
    def z(s):
        std = s.std(ddof=0)
        if not std or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    return frame.groupby(group)[col].transform(z)


def compute_intensity(
    person_weights: pd.DataFrame,
    employers: pd.DataFrame,
    taxonomy: OrgTaxonomy,
    beta: float = BETA,
) -> pd.DataFrame:
    """Per-company intensity with confidence weighting and founder bonus."""
    df = employers.merge(
        person_weights[["author_dblp_pid", "weight"]], on="author_dblp_pid", how="inner"
    )
    df = df[df["employer_org_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=[
            "employer_org_id", "org_type", "is_industry", "group_org_id",
            "intensity_raw", "n_people", "n_founders", "intensity_index", "share", "per_capita",
        ])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.4)
    df["contribution"] = df["weight"] * df["confidence"]
    founder_bonus = beta * (df["is_founder"].astype(bool) * df["weight"])
    df["contribution"] = df["contribution"] + founder_bonus

    grouped = df.groupby("employer_org_id").agg(
        intensity_raw=("contribution", "sum"),
        n_people=("author_dblp_pid", "nunique"),
        n_founders=("is_founder", "sum"),
    ).reset_index()

    grouped["org_type"] = grouped["employer_org_id"].map(taxonomy.org_type)
    grouped["is_industry"] = grouped["employer_org_id"].map(taxonomy.is_industry)
    grouped["group_org_id"] = grouped["employer_org_id"].map(taxonomy.rollup_to_group)

    max_raw = grouped["intensity_raw"].max()
    total_raw = grouped["intensity_raw"].sum()
    grouped["intensity_index"] = 100 * grouped["intensity_raw"] / max_raw if max_raw else 0.0
    grouped["share"] = grouped["intensity_raw"] / total_raw if total_raw else 0.0
    grouped["per_capita"] = grouped["intensity_raw"] / grouped["n_people"].clip(lower=1)
    return grouped.sort_values("intensity_raw", ascending=False).reset_index(drop=True)


def lab_decomposition(
    person_weights: pd.DataFrame,
    employers: pd.DataFrame,
    training: pd.DataFrame,
    beta: float = BETA,
) -> pd.DataFrame:
    """Decompose each company's intensity by the training lab that produced it."""
    df = (
        employers.merge(person_weights[["author_dblp_pid", "weight"]], on="author_dblp_pid")
        .merge(training[["author_dblp_pid", "training_org_id", "advisor_dblp_pid"]],
               on="author_dblp_pid", how="left")
    )
    df = df[df["employer_org_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=[
            "employer_org_id", "training_org_id", "advisor_dblp_pid",
            "contribution", "n_people",
        ])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.4)
    df["contribution"] = df["weight"] * df["confidence"] + beta * (
        df["is_founder"].astype(bool) * df["weight"]
    )
    out = df.groupby(["employer_org_id", "training_org_id", "advisor_dblp_pid"], dropna=False).agg(
        contribution=("contribution", "sum"),
        n_people=("author_dblp_pid", "nunique"),
    ).reset_index()
    return out.sort_values(["employer_org_id", "contribution"], ascending=[True, False])


def run(
    cfg: Dict[str, Any],
    *,
    persons: pd.DataFrame,
    seed_authorships: pd.DataFrame,
    works: pd.DataFrame,
    employers: pd.DataFrame,
    training: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    taxonomy = OrgTaxonomy.from_csv(cfg["paths"]["orgs"])
    weights = compute_person_weights(persons, seed_authorships, works)
    intensity = compute_intensity(weights, employers, taxonomy)
    decomp = (
        lab_decomposition(weights, employers, training)
        if training is not None else pd.DataFrame()
    )

    flow_dir = Path(cfg["paths"]["processed"]) / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    weights[["author_dblp_pid", "display_name", "fractional_citations", "fractional_pubs",
             "venue_family", "weight"]].to_csv(flow_dir / "person_weights.csv", index=False)
    intensity.to_csv(flow_dir / "company_intensity.csv", index=False)
    if not decomp.empty:
        decomp.to_csv(flow_dir / "intensity_lab_decomposition.csv", index=False)
    _write_report(cfg, intensity, decomp, flow_dir)
    LOG.info("stage=intensity companies=%d", len(intensity))
    return {"weights": weights, "intensity": intensity, "decomposition": decomp}


def _write_report(cfg, intensity, decomp, flow_dir) -> None:
    lines = [
        "# AI Talent Intensity Index",
        "",
        "Per-company intensity = confidence-weighted sum of employed talents'",
        f"weights (alpha={ALPHA} citations vs pubs, z-scored within venue family),",
        f"plus a founder bonus (beta={BETA}).",
        "",
        "## Top 25 companies by intensity",
        "",
        "| Rank | Org | Type | Index (0-100) | People | Founders |",
        "|---:|---|---|---:|---:|---:|",
    ]
    top = intensity[intensity["is_industry"]].head(25) if not intensity.empty else intensity
    for i, row in enumerate(top.itertuples(), start=1):
        lines.append(
            f"| {i} | {row.employer_org_id} | {row.org_type} | "
            f"{row.intensity_index:.1f} | {int(row.n_people)} | {int(row.n_founders)} |"
        )
    (flow_dir / "INTENSITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
