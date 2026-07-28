"""Full per-person DBLP history from the bulk XML dump (no per-PID crawl).

The per-PID API (``dblp_person.py``) is correct but slow: ~2s x ~150k authors is
days of wall-clock. This module downloads the monolithic ``dblp.xml.gz`` once and
stream-parses it locally to reconstruct the same history in one pass (minutes).

Two facts drive the design (verified against dblp.dtd):

- The bulk-dump ``<author>`` element has **no pid attribute** — only the
  disambiguated name text (e.g. "Wei Wang 0010") and an optional ``orcid``. So
  co-authorship is keyed by that disambiguated name, and DBLP PIDs are attached
  from the seed's name->pid map (the only source of PIDs is the seed SPARQL).
- ``dblp.xml`` uses custom entities defined in ``dblp.dtd``, so parsing requires
  the DTD alongside the XML with entity resolution enabled (lxml).

Output matches ``dblp_person.HISTORY_COLUMNS`` so the two collectors are
interchangeable behind ``pipeline._collect_history``.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from lxml import etree

from src.collect.dblp_person import HISTORY_COLUMNS

LOG = logging.getLogger("ai_talent_collection.dblp_dump")

_PUB_TAGS = {"article", "inproceedings", "proceedings", "incollection", "book"}
_CHUNK = 1 << 20  # 1 MiB download chunks


def collect(cfg: Dict[str, Any], target_pids: List[str], pid_to_name: Dict[str, str]) -> pd.DataFrame:
    if not cfg.get("dblp_person", {}).get("enabled", True) or not target_pids:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    target_names = {pid_to_name[p] for p in target_pids if p in pid_to_name}
    name_to_pid = {name: pid for pid, name in pid_to_name.items()}
    if not target_names:
        LOG.warning("stage=dblp_dump no_target_names pids=%d", len(target_pids))
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    xml_path = _ensure_dump(cfg)
    rows, matched_pubs = _parse_dump(xml_path, target_names, name_to_pid)
    frame = _finalize(rows)
    LOG.info(
        "stage=dblp_dump target_authors=%d matched_pubs=%d works=%d rows=%d",
        len(target_names),
        matched_pubs,
        frame["dblp_work_id"].nunique() if not frame.empty else 0,
        len(frame),
    )
    return frame


def _parse_dump(xml_path, target_names, name_to_pid):
    """Stream-parse the dump, emitting authorship rows for pubs touching a target."""
    rows: List[Dict[str, Any]] = []
    retrieved_at = pd.Timestamp.utcnow().isoformat()
    matched_pubs = 0
    context = etree.iterparse(
        str(xml_path),
        events=("end",),
        load_dtd=True,
        resolve_entities=True,
        huge_tree=True,
        recover=True,
    )
    for _event, elem in context:
        # Only act on record 'end' events. Clearing non-record elements would
        # delete a pub's <author>/<year> children before its own end event fires.
        if elem.tag not in _PUB_TAGS:
            continue
        authors = [
            (a.text.strip(), a.get("orcid"))
            for a in elem.findall("author")
            if a.text and a.text.strip()
        ]
        names = [n for n, _ in authors]
        if authors and not target_names.isdisjoint(names):
            matched_pubs += 1
            key = elem.get("key")
            if key:
                work_id = f"https://dblp.org/rec/{key}"
                year = _text_int(elem.find("year"))
                venue = _first_text(elem, ("booktitle", "journal"))
                for ordinal, (name, orc) in enumerate(authors, start=1):
                    rows.append({
                        "dblp_work_id": work_id,
                        "author_dblp_pid": name_to_pid.get(name),
                        "author_name": name,
                        "author_orcid": orc,
                        "author_ordinal": ordinal,
                        "publication_year": year,
                        "venue_raw": venue,
                        "retrieved_at": retrieved_at,
                        "source": "dblp_dump",
                    })
        _clear(elem)
    del context
    return rows, matched_pubs


def _finalize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.DataFrame(rows)
    # PID may be None for non-seed co-authors, so dedupe on (work, name).
    frame = frame.drop_duplicates(["dblp_work_id", "author_name"])
    max_ord = frame.groupby("dblp_work_id")["author_ordinal"].transform("max")
    position = pd.Series("middle", index=frame.index)
    position = position.mask(frame["author_ordinal"] == 1, "first")
    position = position.mask((frame["author_ordinal"] == max_ord) & (max_ord > 1), "last")
    frame["author_position"] = position
    return frame[HISTORY_COLUMNS].reset_index(drop=True)


def _ensure_dump(cfg: Dict[str, Any]) -> Path:
    """Download dblp.xml.gz + dblp.dtd once and decompress to a local file."""
    external = Path(cfg["paths"]["external"])
    external.mkdir(parents=True, exist_ok=True)
    dp = cfg.get("dblp_person", {})
    gz_path = external / "dblp.xml.gz"
    xml_path = external / "dblp.xml"
    dtd_path = external / "dblp.dtd"

    ua = cfg.get("_user_agent", "ai-talent-collection")
    if not dtd_path.exists():
        _download(dp.get("dtd_url", "https://dblp.org/xml/dblp.dtd"), dtd_path, ua)
    if not xml_path.exists():
        if not gz_path.exists():
            _download(dp.get("dump_url", "https://dblp.org/xml/dblp.xml.gz"), gz_path, ua)
        LOG.info("stage=dblp_dump decompressing %s -> %s", gz_path, xml_path)
        with gzip.open(gz_path, "rb") as src, open(xml_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK)
    return xml_path


def _download(url: str, dest: Path, user_agent: str) -> None:
    LOG.info("stage=dblp_dump downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, headers={"User-Agent": user_agent}, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def _clear(elem) -> None:
    """Release memory for a processed element and its already-parsed siblings."""
    elem.clear()
    parent = elem.getparent()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]


def _first_text(element, tags) -> Optional[str]:
    for tag in tags:
        found = element.find(tag)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return None


def _text_int(element) -> Optional[int]:
    if element is None or not (element.text or "").strip():
        return None
    try:
        return int(element.text.strip())
    except ValueError:
        return None
