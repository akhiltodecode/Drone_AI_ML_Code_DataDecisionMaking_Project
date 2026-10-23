#!/usr/bin/env python3
"""Independent structural and checksum verification for the extended study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    state = json.loads((output_dir / "extended_run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["replications_completed"] == state["replications_requested"]
    replications = int(state["replications_completed"])
    folds = int(state["cv_folds"])
    candidates = int(state["search_candidates"])

    trials = pd.read_csv(output_dir / "tuning_trials.csv")
    expected_trials = replications * 3 * candidates * folds
    assert len(trials) == expected_trials, (len(trials), expected_trials)
    assert trials["replicate"].nunique() == replications
    assert trials["family"].nunique() == 3
    assert trials["brier"].between(0, 1).all()
    assert trials["pr_auc"].between(0, 1).all()

    selected = pd.read_csv(output_dir / "selected_hyperparameters.csv")
    assert len(selected) == replications * 3
    assert selected.groupby("replicate")["family"].nunique().eq(3).all()

    extended = pd.read_csv(output_dir / "extended_results.csv")
    assert len(extended) == replications
    assert extended[["brier", "pr_auc"]].apply(lambda column: column.between(0, 1)).all().all()

    ablations = pd.read_csv(output_dir / "ablation_results.csv")
    assert len(ablations) == replications * 4
    assert ablations.groupby("replicate")["ablation"].nunique().eq(4).all()

    checked = 0
    manifest = output_dir / "extended_checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        assert sha256(output_dir / filename) == expected, filename
        checked += 1

    print(f"PASS: {replications} extended replications")
    print(f"PASS: {expected_trials} leakage-safe grouped-CV trials")
    print("PASS: three selected model families and four modality ablations")
    print(f"PASS: {checked} extended scientific checksums verified")


if __name__ == "__main__":
    main()
