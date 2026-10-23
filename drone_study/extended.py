"""Nested, grouped model-selection study for prolonged RunPod execution.

The outer Monte Carlo replications remain independent. Within each replication,
hyperparameters are selected only from the training assets using deterministic
grouped cross-validation. Probability calibration uses the separate calibration
assets and final metrics use the untouched test assets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .config import CONFIG, CONFIG_PATH, CONFIG_SHA256, PROJECT_ROOT
from .pipeline import (
    CAT_COLS,
    DATA_SEEDS,
    FEATURE_COLS,
    MODEL_SEEDS,
    NUM_COLS,
    SigmoidCalibrator,
    generate_replication,
    grouped_split,
    optimal_actions,
    realized_expected_loss,
)
from .runtime import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    cleanup_stale_temporary_files,
    configure_logging,
    read_checkpoint,
    runtime_metadata,
    write_checkpoint,
)


EXTENDED = CONFIG["extended_study"]
FAMILIES = tuple(EXTENDED["model_families"])
RGB_FEATURES = {"rgb_corrosion", "rgb_crack", "rgb_coating"}
THERMAL_FEATURES = {"thermal_delta", "thermal_hotspot", "thermal_gradient"}
FLIGHT_FEATURES = {
    "wind_ms", "standoff_m", "illumination", "blur",
    "thermal_confidence", "flight_distance_m", "battery_remaining",
}
ABLATIONS = {
    "all_features": set(),
    "without_rgb": RGB_FEATURES,
    "without_thermal": THERMAL_FEATURES,
    "without_flight_quality": FLIGHT_FEATURES,
}


def candidate_grid(family: str) -> list[dict[str, Any]]:
    """Return the 12 prespecified candidates for one model family."""
    if family == "logistic_regression":
        return [
            {"C": c, "class_weight": weight}
            for c in (0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
            for weight in ("balanced", None)
        ]
    if family == "random_forest":
        return [
            {
                "n_estimators": trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": "balanced",
            }
            for trees in (200, 400, 600)
            for depth in (6, 10)
            for leaf in (5, 15)
        ]
    if family == "hist_gradient_boosting":
        return [
            {
                "max_iter": 200,
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "l2_regularization": regularization,
            }
            for rate in (0.03, 0.06, 0.10)
            for leaves in (15, 31)
            for regularization in (0.5, 2.0)
        ]
    raise ValueError(f"unknown model family: {family}")


def grouped_folds(groups: pd.Series, folds: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic validation folds with complete asset separation."""
    unique = np.asarray(pd.unique(groups), dtype=object)
    if folds > len(unique):
        raise ValueError("cv folds exceed the number of training assets")
    rng = np.random.default_rng(seed + 510_007)
    rng.shuffle(unique)
    group_values = groups.to_numpy()
    for validation_groups in np.array_split(unique, folds):
        validation = np.isin(group_values, validation_groups)
        train_index = np.flatnonzero(~validation)
        validation_index = np.flatnonzero(validation)
        if not len(train_index) or not len(validation_index):
            raise ValueError("empty grouped cross-validation fold")
        yield train_index, validation_index


def feature_columns(ablation: str) -> list[str]:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    return [column for column in FEATURE_COLS if column not in ABLATIONS[ablation]]


def feature_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    numeric = [column for column in NUM_COLS if column in features]
    categorical = [column for column in CAT_COLS if column in features]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def build_model(
    family: str,
    parameters: dict[str, Any],
    seed: int,
    features: list[str],
) -> Pipeline:
    if family == "logistic_regression":
        estimator = LogisticRegression(
            C=float(parameters["C"]),
            class_weight=parameters["class_weight"],
            max_iter=2500,
            solver="lbfgs",
        )
        scale = True
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            class_weight=parameters["class_weight"],
            random_state=seed,
            n_jobs=1,
        )
        scale = False
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=int(parameters["max_iter"]),
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            random_state=seed,
        )
        scale = False
    else:
        raise ValueError(f"unknown model family: {family}")
    return Pipeline([
        ("prep", feature_preprocessor(features, scale)),
        ("model", estimator),
    ])


def fit_model(model: Pipeline, family: str, frame: pd.DataFrame, features: list[str]) -> Pipeline:
    target = frame["failure_12m"].to_numpy()
    if family == "hist_gradient_boosting":
        weights = compute_sample_weight("balanced", target)
        model.fit(frame[features], target, model__sample_weight=weights)
    else:
        model.fit(frame[features], target)
    return model


def observed_decision_loss(
    outcome: np.ndarray,
    probability: np.ndarray,
    criticality: pd.Series,
) -> float:
    """Empirical decision loss using observed outcomes, never simulator truth."""
    actions = optimal_actions(probability, criticality)
    return float(realized_expected_loss(outcome.astype(float), actions, criticality).mean())


def evaluate_candidates(
    train: pd.DataFrame,
    family: str,
    seed: int,
    cv_folds: int,
    search_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = feature_columns("all_features")
    folds = list(grouped_folds(train["asset_id"], cv_folds, seed))
    rows: list[dict[str, Any]] = []
    candidates = candidate_grid(family)[:search_candidates]
    for candidate_index, parameters in enumerate(candidates, start=1):
        candidate_id = f"{family}_{candidate_index:02d}"
        parameter_json = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        for fold_index, (fit_index, validation_index) in enumerate(folds, start=1):
            fit_frame = train.iloc[fit_index]
            validation = train.iloc[validation_index]
            fit_assets = set(fit_frame["asset_id"])
            validation_assets = set(validation["asset_id"])
            if not fit_assets.isdisjoint(validation_assets):
                raise AssertionError("asset leakage in grouped cross-validation")
            model = build_model(family, parameters, seed + fold_index, features)
            fit_model(model, family, fit_frame, features)
            probability = np.clip(
                model.predict_proba(validation[features])[:, 1], 1e-6, 1 - 1e-6
            )
            outcome = validation["failure_12m"].to_numpy()
            rows.append({
                "family": family,
                "candidate_id": candidate_id,
                "parameters": parameter_json,
                "fold": fold_index,
                "training_assets": len(fit_assets),
                "validation_assets": len(validation_assets),
                "brier": float(brier_score_loss(outcome, probability)),
                "pr_auc": float(average_precision_score(outcome, probability)),
                "observed_decision_loss": observed_decision_loss(
                    outcome, probability, validation["criticality"]
                ),
            })
    trial_frame = pd.DataFrame(rows)
    summary = (
        trial_frame.groupby(["candidate_id", "parameters"], as_index=False)
        .agg(mean_brier=("brier", "mean"), mean_pr_auc=("pr_auc", "mean"))
        .sort_values(["mean_brier", "mean_pr_auc", "candidate_id"], ascending=[True, False, True])
    )
    winner = summary.iloc[0]
    selected = {
        "family": family,
        "candidate_id": winner["candidate_id"],
        "parameters": json.loads(winner["parameters"]),
        "mean_cv_brier": float(winner["mean_brier"]),
        "mean_cv_pr_auc": float(winner["mean_pr_auc"]),
    }
    return rows, selected


def held_out_metrics(
    test: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, float]:
    outcome = test["failure_12m"].to_numpy()
    true_probability = test["true_probability"].to_numpy()
    action = optimal_actions(probability, test["criticality"])
    oracle_action = optimal_actions(true_probability, test["criticality"])
    losses = realized_expected_loss(true_probability, action, test["criticality"])
    oracle_losses = realized_expected_loss(
        true_probability, oracle_action, test["criticality"]
    )
    return {
        "brier": float(brier_score_loss(outcome, probability)),
        "pr_auc": float(average_precision_score(outcome, probability)),
        "mel": float(losses.mean()),
        "regret": float((losses - oracle_losses).mean()),
    }


def fit_calibrate_evaluate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    seed: int,
    ablation: str,
) -> dict[str, float]:
    features = feature_columns(ablation)
    model = build_model(family, parameters, seed, features)
    fit_model(model, family, train, features)
    calibration_base = model.predict_proba(calibration[features])[:, 1]
    calibrator = SigmoidCalibrator().fit(
        calibration_base, calibration["failure_12m"].to_numpy()
    )
    test_base = model.predict_proba(test[features])[:, 1]
    probability = calibrator.predict(test_base)
    return held_out_metrics(test, probability)


def execute_extended_replication(
    replicate: int,
    data_seed: int,
    model_seed: int,
    cv_folds: int,
    search_candidates: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    data = generate_replication(data_seed, replicate)
    train, calibration, test = grouped_split(data, data_seed)
    tuning_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        rows, selected = evaluate_candidates(
            train, family, model_seed, cv_folds, search_candidates
        )
        for row in rows:
            tuning_rows.append({"replicate": replicate, **row})
        selected_rows.append({
            "replicate": replicate,
            "family": family,
            "candidate_id": selected["candidate_id"],
            "parameters": json.dumps(selected["parameters"], sort_keys=True),
            "mean_cv_brier": selected["mean_cv_brier"],
            "mean_cv_pr_auc": selected["mean_cv_pr_auc"],
        })

    best = sorted(
        selected_rows,
        key=lambda row: (row["mean_cv_brier"], -row["mean_cv_pr_auc"], row["family"]),
    )[0]
    best_parameters = json.loads(best["parameters"])
    result_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for ablation in EXTENDED["ablations"]:
        metrics = fit_calibrate_evaluate(
            train,
            calibration,
            test,
            best["family"],
            best_parameters,
            model_seed,
            ablation,
        )
        record = {
            "replicate": replicate,
            "selected_family": best["family"],
            "selected_candidate_id": best["candidate_id"],
            "ablation": ablation,
            **metrics,
        }
        ablation_rows.append(record)
        if ablation == "all_features":
            result_rows.append(record.copy())

    signature = {
        "protocol_version": CONFIG["protocol_version"],
        "config_sha256": CONFIG_SHA256,
        "cv_folds": cv_folds,
        "search_candidates": search_candidates,
    }
    return {
        "replicate": replicate,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "signature": signature,
        "tuning_rows": tuning_rows,
        "selected_rows": selected_rows,
        "result_rows": result_rows,
        "ablation_rows": ablation_rows,
        "resource_rows": [{
            "replicate": replicate,
            "elapsed_seconds": time.perf_counter() - started_at,
            "candidate_fits": len(FAMILIES) * search_candidates * cv_folds,
            "final_fits": len(EXTENDED["ablations"]),
        }],
    }


def _execute_and_checkpoint(task: tuple[int, int, int, int, int, str]) -> dict[str, Any]:
    replicate, data_seed, model_seed, cv_folds, search_candidates, checkpoint = task
    with threadpool_limits(limits=1):
        bundle = execute_extended_replication(
            replicate, data_seed, model_seed, cv_folds, search_candidates
        )
    write_checkpoint(Path(checkpoint), bundle)
    return bundle


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(bundles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bundle in sorted(bundles, key=lambda item: item["replicate"]):
        records.extend(bundle[key])
    return records


def run_extended_study(
    output_dir: Path,
    replications: int,
    cv_folds: int,
    search_candidates: int,
    workers: int,
    resume: bool,
    verbose: bool = False,
) -> None:
    logger = configure_logging(verbose)
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "extended_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_temporary_files(output_dir)
    cleanup_stale_temporary_files(checkpoint_dir)

    expected_signature = {
        "protocol_version": CONFIG["protocol_version"],
        "config_sha256": CONFIG_SHA256,
        "cv_folds": cv_folds,
        "search_candidates": search_candidates,
    }
    state = {
        "status": "running",
        **expected_signature,
        "replications_requested": replications,
        "workers": workers,
        "resume": resume,
        "fits_planned": replications * (
            len(FAMILIES) * search_candidates * cv_folds + len(EXTENDED["ablations"])
        ),
    }
    atomic_write_json(output_dir / "extended_run_state.json", state)

    bundles: list[dict[str, Any]] = []
    pending: list[tuple[int, int, int, int, int, str]] = []
    selected_seeds = zip(DATA_SEEDS[:replications], MODEL_SEEDS[:replications])
    for replicate, (data_seed, model_seed) in enumerate(selected_seeds, start=1):
        checkpoint = checkpoint_dir / f"extended_replication_{replicate:02d}.json.gz"
        if resume and checkpoint.is_file():
            bundle = read_checkpoint(checkpoint, replicate, data_seed, model_seed)
            if bundle.get("signature") != expected_signature:
                raise ValueError(f"extended checkpoint protocol mismatch: {checkpoint}")
            bundles.append(bundle)
            logger.info("extended replication %02d restored from checkpoint", replicate)
        else:
            pending.append((
                replicate,
                data_seed,
                model_seed,
                cv_folds,
                search_candidates,
                str(checkpoint),
            ))

    if pending:
        logger.info(
            "executing %d extended replication(s) with %d worker process(es)",
            len(pending),
            workers,
        )
        if workers == 1:
            for task in pending:
                bundle = _execute_and_checkpoint(task)
                bundles.append(bundle)
                logger.info("extended replication %02d completed", bundle["replicate"])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_execute_and_checkpoint, task): task[0]
                    for task in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    bundle = future.result()
                    bundles.append(bundle)
                    logger.info("extended replication %02d completed", bundle["replicate"])

    bundles.sort(key=lambda item: item["replicate"])
    artifacts = {
        "tuning_trials.csv": pd.DataFrame(_flatten(bundles, "tuning_rows")),
        "selected_hyperparameters.csv": pd.DataFrame(_flatten(bundles, "selected_rows")),
        "extended_results.csv": pd.DataFrame(_flatten(bundles, "result_rows")),
        "ablation_results.csv": pd.DataFrame(_flatten(bundles, "ablation_rows")),
        "resource_usage.csv": pd.DataFrame(_flatten(bundles, "resource_rows")),
    }
    for filename, frame in artifacts.items():
        atomic_write_csv(frame, output_dir / filename)

    provenance = {
        "protocol_version": CONFIG["protocol_version"],
        "config_path": CONFIG_PATH.name,
        "config_sha256": CONFIG_SHA256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "data_seeds": DATA_SEEDS[:replications],
        "model_seeds": MODEL_SEEDS[:replications],
        "cv_folds": cv_folds,
        "candidates_per_family": search_candidates,
        "selection_metric": EXTENDED["selection_metric"],
        "pythonhashseed": os.environ.get(
            "PYTHONHASHSEED", "not set; all numerical RNGs explicitly seeded"
        ),
    }
    atomic_write_json(output_dir / "extended_environment.json", provenance)

    scientific_files = [
        "tuning_trials.csv",
        "selected_hyperparameters.csv",
        "extended_results.csv",
        "ablation_results.csv",
        "extended_environment.json",
    ]
    manifest = [
        f"{_file_sha256(output_dir / filename)}  {filename}"
        for filename in scientific_files
    ]
    atomic_write_text(
        output_dir / "extended_checksums.sha256", "\n".join(manifest) + "\n"
    )
    state.update(runtime_metadata(started_at, PROJECT_ROOT))
    state["status"] = "completed"
    state["replications_completed"] = len(bundles)
    atomic_write_json(output_dir / "extended_run_state.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grouped nested model selection for the synthetic drone study."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--replications", type=int, default=len(DATA_SEEDS))
    parser.add_argument("--cv-folds", type=int, default=int(EXTENDED["cv_folds"]))
    parser.add_argument(
        "--search-candidates",
        type=int,
        default=int(EXTENDED["candidates_per_model_family"]),
        help="prespecified candidates evaluated per model family (1-12)",
    )
    parser.add_argument("--workers", "--threads", dest="workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.replications <= len(DATA_SEEDS):
        parser.error(f"--replications must be between 1 and {len(DATA_SEEDS)}")
    if not 2 <= args.cv_folds <= int(CONFIG["study"]["train_assets"]):
        parser.error("--cv-folds must be between 2 and the number of training assets")
    if not 1 <= args.search_candidates <= 12:
        parser.error("--search-candidates must be between 1 and 12")
    if not 1 <= args.workers <= args.replications:
        parser.error("--workers must be between 1 and --replications")
    run_extended_study(
        args.output_dir.resolve(),
        args.replications,
        args.cv_folds,
        args.search_candidates,
        args.workers,
        args.resume,
        args.verbose,
    )


if __name__ == "__main__":
    main()
