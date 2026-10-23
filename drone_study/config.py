"""Load and validate the locked, human-readable study configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "study_config.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read and validate the complete protocol configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    required = {
        "protocol_version", "study", "seeds", "decision_model",
        "robustness", "extended_study", "publication_limits",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"study configuration is missing: {sorted(missing)}")

    study = config["study"]
    split_total = (
        study["train_assets"]
        + study["calibration_assets"]
        + study["test_assets"]
    )
    if split_total != study["assets_per_replication"]:
        raise ValueError("asset split does not equal assets_per_replication")

    data_seeds = config["seeds"]["data"]
    model_seeds = config["seeds"]["model"]
    if len(data_seeds) != len(model_seeds) or len(set(data_seeds)) != len(data_seeds):
        raise ValueError("data/model seeds must be unique paired lists of equal length")
    if not 1 <= study["default_replications"] <= len(data_seeds):
        raise ValueError("default_replications is outside the prespecified seed list")
    if study["bootstrap_resamples"] < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")

    extended = config["extended_study"]
    if not 2 <= extended["cv_folds"] <= study["train_assets"]:
        raise ValueError("extended-study cv_folds is outside the grouped training split")
    if not 1 <= extended["candidates_per_model_family"] <= 12:
        raise ValueError("extended-study candidates_per_model_family must be between 1 and 12")
    if extended["selection_metric"] != "grouped_cv_brier_score":
        raise ValueError("the extended-study selection metric must remain grouped CV Brier score")
    expected_families = {
        "logistic_regression", "random_forest", "hist_gradient_boosting",
    }
    if set(extended["model_families"]) != expected_families:
        raise ValueError("extended-study model families do not match the locked protocol")

    decision = config["decision_model"]
    action_count = len(decision["actions"])
    if action_count != 4:
        raise ValueError("the locked protocol requires exactly four actions")
    if len(decision["fixed_probability_thresholds"]) != action_count - 1:
        raise ValueError("fixed thresholds must separate every adjacent action")
    if sorted(decision["fixed_probability_thresholds"]) != decision["fixed_probability_thresholds"]:
        raise ValueError("fixed thresholds must be increasing")
    if len(decision["residual_failure_fractions"]) != action_count:
        raise ValueError("one residual failure fraction is required per action")
    for criticality in ("low", "medium", "high"):
        if len(decision["direct_action_cost"][criticality]) != action_count:
            raise ValueError(f"invalid direct-action cost vector for {criticality}")

    limits = config["publication_limits"]
    if limits != {"maximum_paper_figures": 2, "maximum_paper_tables": 2}:
        raise ValueError("this short-communication protocol is locked to two figures and two tables")
    return config


CONFIG = load_config()
CONFIG_SHA256 = _sha256(CONFIG_PATH)
