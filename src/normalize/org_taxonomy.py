"""Single source of truth for organization-type classification and rollups.

The curated ``data/orgs.csv`` uses several ``org_type`` values. Historically the
flow code classified industry destinations with
``org_type.isin(['industry_lab', 'industry_parent'])`` — which silently drops
every row typed plain ``industry`` (the large majority of company rows in the
current file). This module centralizes the taxonomy so that bug cannot recur in
more than one place.

Rollup semantics (see the ``notes`` column in ``data/orgs.csv``):

- ``industry_parent`` rows (Alphabet, Meta, Microsoft, ByteDance, Alibaba) are
  *shells*. They must never be a flow destination in their own right — an
  affiliation should resolve to the ``industry_lab`` child (e.g.
  ``google_deepmind``), and the parent exists only so children can roll up for a
  corporate-group view.
- ``rollup_to_group`` walks ``parent_org_id`` to that group view; the default
  destination granularity is the leaf lab.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

# Every org_type that represents a company / for-profit lab.
INDUSTRY_TYPES = {"industry", "industry_lab", "industry_parent"}
# Universities. OpenAlex's legacy "education" value is intentionally not
# included — OpenAlex is no longer a source in this pipeline.
ACADEMIC_TYPES = {"academic"}
# Types that are neither company nor university but are still real employers.
OTHER_EMPLOYER_TYPES = {"nonprofit", "government", "healthcare", "facility"}

# Shell parents may never be a leaf destination on their own.
SHELL_TYPES = {"industry_parent"}


def is_industry(org_type: Optional[str]) -> bool:
    return (org_type or "") in INDUSTRY_TYPES


def is_academic(org_type: Optional[str]) -> bool:
    return (org_type or "") in ACADEMIC_TYPES


def is_shell(org_type: Optional[str]) -> bool:
    """A shell parent exists only for group rollups, never as a destination."""
    return (org_type or "") in SHELL_TYPES


def is_valid_destination(org_type: Optional[str]) -> bool:
    """True for org types that may appear as a flow destination / employer."""
    t = org_type or ""
    if not t:
        return False
    if is_shell(t):
        return False
    return t in INDUSTRY_TYPES or t in ACADEMIC_TYPES or t in OTHER_EMPLOYER_TYPES


class OrgTaxonomy:
    """Loads ``orgs.csv`` once and answers type / rollup questions by ``org_id``."""

    def __init__(self, org_type_by_id: Dict[str, str], parent_by_id: Dict[str, str]):
        self._org_type = org_type_by_id
        self._parent = parent_by_id

    @classmethod
    def from_csv(cls, org_path: str | Path) -> "OrgTaxonomy":
        path = Path(org_path)
        org_type_by_id: Dict[str, str] = {}
        parent_by_id: Dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                org_id = (row.get("org_id") or "").strip()
                if not org_id:
                    continue
                org_type_by_id[org_id] = (row.get("org_type") or "").strip()
                parent = (row.get("parent_org_id") or "").strip()
                if parent:
                    parent_by_id[org_id] = parent
        return cls(org_type_by_id, parent_by_id)

    def org_type(self, org_id: Optional[str]) -> Optional[str]:
        if org_id is None:
            return None
        return self._org_type.get(org_id)

    def is_industry(self, org_id: Optional[str]) -> bool:
        return is_industry(self.org_type(org_id))

    def is_academic(self, org_id: Optional[str]) -> bool:
        return is_academic(self.org_type(org_id))

    def is_valid_destination(self, org_id: Optional[str]) -> bool:
        return is_valid_destination(self.org_type(org_id))

    def rollup_to_group(self, org_id: Optional[str]) -> Optional[str]:
        """Return the top-level group id (walk ``parent_org_id``), else the id itself."""
        if org_id is None:
            return None
        seen = set()
        current = org_id
        while current in self._parent and current not in seen:
            seen.add(current)
            current = self._parent[current]
        return current
