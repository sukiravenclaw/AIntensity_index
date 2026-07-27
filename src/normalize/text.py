from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"\bthe\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_title(value: Any) -> str:
    return normalize_text(value)


def normalize_doi(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", text, flags=re.I)
    return text.casefold() or None


def normalize_ror(value: Any) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"(0[a-hj-km-np-tv-z0-9]{6}\d{2})", str(value), re.I)
    if match:
        return f"https://ror.org/{match.group(1).lower()}"
    return str(value).strip().rstrip("/").casefold() or None

