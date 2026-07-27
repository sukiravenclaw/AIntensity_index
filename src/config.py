from __future__ import annotations

import os
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigurationError(ValueError):
    pass


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent)

    end = cfg.get("window", {}).get("end")
    if not end:
        cfg["window"]["end"] = date.today().isoformat()

    for key, value in cfg.get("paths", {}).items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        cfg["paths"][key] = str(candidate.resolve())

    project = cfg.get("project", {})
    contact_env = project.get("contact_email_env", "CONTACT_EMAIL")
    contact = os.getenv(contact_env, "").strip()
    if not contact or "@" not in contact:
        raise ConfigurationError(
            f"Set {contact_env} to a monitored contact email; it is required "
            "for the descriptive User-Agent."
        )
    cfg["_contact_email"] = contact
    cfg["_user_agent"] = f"{project.get('user_agent', 'ai-talent-collection/0.1')} (mailto:{contact})"

    openalex = cfg.get("openalex", {})
    key_env = openalex.get("api_key_env", "OPENALEX_API_KEY")
    cfg["_openalex_api_key"] = os.getenv(key_env, "").strip()
    if openalex.get("enabled", True) and openalex.get("require_api_key", True) and not cfg["_openalex_api_key"]:
        raise ConfigurationError(
            f"Set {key_env}. OpenAlex's current API documentation requires a "
            "key for practical list-query throughput."
        )

    s2 = cfg.get("semantic_scholar", {})
    cfg["_s2_api_key"] = os.getenv(s2.get("api_key_env", "SEMANTIC_SCHOLAR_API_KEY"), "").strip()
    return cfg


def ensure_directories(cfg: Dict[str, Any]) -> None:
    for key in ("raw", "external", "interim", "processed"):
        if key in cfg["paths"]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

