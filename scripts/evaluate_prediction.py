"""Evaluate the Intelligence Index prediction function reported in Table 9 (RQ5).

The implementation follows Kim et al.'s 19-predictor OLS specification and
feature construction. It refits the function on Kim et al.'s 260 configurations
using Artificial Analysis Intelligence Index v4.1 as model capability.

It then evaluates the fitted functions, without refitting, on the mean success
rates of the homogeneous RQ2 runs (six models; the Gemini 2.5 Flash runs of RQ1
are excluded). Only the R-squared values needed by
the manuscript are printed. The data directory is deliberately supplied as a
command-line argument so source and data can be distributed as separate ZIPs.

Usage:
    uv run python scripts/evaluate_prediction.py --data-dir data
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


COORDINATION_METRICS = {
    "single-agent": (0.0, 0.00, 0.00, 0.466, 1.0),
    "independent": (58.0, 0.00, 0.48, 0.234, 17.2),
    "decentralized": (263.0, 0.41, 0.50, 0.132, 7.8),
    "centralized": (285.0, 0.39, 0.41, 0.120, 4.4),
    "hybrid": (515.0, 0.24, 0.46, 0.074, 5.1),
}

TOOL_COUNTS = {
    "BrowseComp-Plus": 3,
    "Finance Agent": 5,
    "PlanCraft": 4,
    "WorkBench": 16,
    "SWE-Bench Verified": 7,
    "Terminal-Bench": 2,
}

AGENT_COUNTS = {
    "single-agent": 1,
    "independent": 3,
    "decentralized": 3,
    "centralized": 4,
    "hybrid": 4,
}

TRANSFER_MODELS = {
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemma-4-31b",
    "gemma-4-e4b",
    "qwen3-5-397b",
    "qwen3-5-9b",
}

BASE_FEATURES = [
    "log_tools",
    "log_agents",
    "log_overhead",
    "message_density",
    "redundancy",
    "efficiency",
    "log_error_amp",
    "single_agent_baseline",
]

INTERACTION_FEATURES = [
    "capability_x_efficiency",
    "error_x_baseline",
    "overhead_x_tools",
    "redundancy_x_agents",
    "message_density_x_capability",
    "efficiency_x_tools",
    "baseline_x_agents",
    "capability_x_tools",
    "error_x_tools",
]

# Kim et al.'s notation is retained here as an explicit mapping to the readable
# column names used below: I is model capability, T is tool count, n_a is agent
# count, O% is overhead, c is
# message density, R is redundancy, E_c is efficiency, and A_e^trace is error
# amplification. FEATURES implements beta_1 through beta_19 in Equation 1.
KIM_PARAMETERS = {
    "I": "capability_centered",
    "T": "n_tools",
    "n_a": "n_agents",
    "O%": "overhead_pct",
    "c": "message_density",
    "R": "redundancy",
    "E_c": "efficiency",
    "A_e^trace": "error_amplification",
    "P_SA": "single_agent_baseline",
}

FEATURES = ["capability_centered", "capability_sq_centered"] + BASE_FEATURES + INTERACTION_FEATURES


def canonical_model(value: str) -> str:
    """Map repository and Kim model spellings to one stable join key."""
    key = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    key = re.sub(r"-20\d{6}$", "", key)
    aliases = {
        "claude-3-7-sonnet": "claude-3-7-sonnet",
        "claude-sonnet-4": "claude-sonnet-4",
        "claude-sonnet-4-5": "claude-sonnet-4-5",
        "gemini-2-0-flash": "gemini-2-0-flash",
        "gemini-2-5-flash": "gemini-2-5-flash",
        "gemini-2-5-pro": "gemini-2-5-pro",
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        "deepseek-v4-flash": "deepseek-v4-flash",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "qwen3-5-397b": "qwen3-5-397b",
        "qwen3-5-9b": "qwen3-5-9b",
        "gemma-4-31b": "gemma-4-31b",
        "gemma-4-e4b": "gemma-4-e4b",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported model name: {value!r}")
    return aliases[key]


def canonical_architecture(value: str) -> str:
    key = str(value).lower().replace("_", "-")
    key = key.removeprefix("multi-agent-")
    if key not in AGENT_COUNTS:
        raise ValueError(f"Unsupported architecture: {value!r}")
    return key


def canonical_benchmark(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    aliases = {
        "browsecompplus": "BrowseComp-Plus",
        "browsecompplussampled100": "BrowseComp-Plus",
        "financeagent": "Finance Agent",
        "plancraft": "PlanCraft",
        "plancrafttest": "PlanCraft",
        "workbench": "WorkBench",
        "swebenchverified": "SWE-Bench Verified",
        "terminalbench": "Terminal-Bench",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported benchmark: {value!r}")
    return aliases[key]


def load_training(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"benchmark", "model", "architecture", "performance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["performance"] = pd.to_numeric(frame["performance"], errors="raise")
    return normalize_configurations(frame, path)


def load_test_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    required = {"benchmark", "model", "architecture", "run", "success_rate"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    group_keys = ["benchmark", "model", "architecture"]
    duplicate_runs = runs.duplicated(group_keys + ["run"])
    if duplicate_runs.any():
        raise ValueError(f"{path} contains duplicate repetition identifiers")

    counts = runs.groupby(group_keys)["run"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Homogeneous configurations do not all have the same repetition count")

    frame = (
        runs.groupby(group_keys, as_index=False)
        .agg(performance=("success_rate", "mean"))
    )
    frame = normalize_configurations(frame, path)
    return frame[frame["model"].isin(TRANSFER_MODELS)].reset_index(drop=True)


def normalize_configurations(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = frame.copy()
    result["benchmark"] = result["benchmark"].map(canonical_benchmark)
    result["model"] = result["model"].map(canonical_model)
    result["architecture"] = result["architecture"].map(canonical_architecture)
    keys = ["benchmark", "model", "architecture"]
    if result.duplicated(keys).any():
        raise ValueError(f"{source} has duplicate configuration rows")
    if not result["performance"].between(0.0, 1.0).all():
        raise ValueError(f"{source} has performance values outside [0, 1]")
    # Do not sort: Kim et al.'s shuffled K-fold split is seeded but still
    # depends on the input row order. The supplied training CSV preserves the
    # order used by their analysis.
    return result.reset_index(drop=True)


def load_intelligence(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"model", "intelligence_index", "index_version"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    versions = {str(value) for value in frame["index_version"]}
    if versions != {"4.1"}:
        raise ValueError(f"Expected only Intelligence Index v4.1, found {sorted(versions)}")
    frame["model_key"] = frame["model"].map(canonical_model)
    if frame["model_key"].duplicated().any():
        raise ValueError(f"{path} has duplicate model scores")
    return dict(zip(frame["model_key"], frame["intelligence_index"], strict=True))


def add_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["n_tools"] = result["benchmark"].map(TOOL_COUNTS)
    result["n_agents"] = result["architecture"].map(AGENT_COUNTS)

    metrics = result["architecture"].map(COORDINATION_METRICS)
    metric_frame = pd.DataFrame(
        metrics.tolist(),
        columns=["overhead_pct", "message_density", "redundancy", "efficiency", "error_amplification"],
        index=result.index,
    )
    result = pd.concat([result, metric_frame], axis=1)

    baselines = (
        result[result["architecture"] == "single-agent"]
        .set_index(["benchmark", "model"])["performance"]
    )
    result["single_agent_baseline"] = [
        baselines.get((benchmark, model), np.nan)
        for benchmark, model in zip(result["benchmark"], result["model"], strict=True)
    ]
    if result["single_agent_baseline"].isna().any():
        missing = result.loc[result["single_agent_baseline"].isna(), ["benchmark", "model"]]
        raise ValueError(f"Missing Single-Agent baselines:\n{missing.drop_duplicates().to_string(index=False)}")

    # Kim's script uses the dataset-level Single-Agent mean for SAS rows.
    dataset_means = (
        result[result["architecture"] == "single-agent"]
        .groupby("benchmark")["performance"]
        .mean()
    )
    sas = result["architecture"] == "single-agent"
    result.loc[sas, "single_agent_baseline"] = result.loc[sas, "benchmark"].map(dataset_means)

    result["log_tools"] = np.log1p(result["n_tools"])
    result["log_agents"] = np.log1p(result["n_agents"])
    result["log_overhead"] = np.log1p(result["overhead_pct"])
    result["log_error_amp"] = np.log1p(result["error_amplification"])
    result["error_x_baseline"] = result["error_amplification"] * result["single_agent_baseline"]
    result["overhead_x_tools"] = result["overhead_pct"] * result["n_tools"]
    result["redundancy_x_agents"] = result["redundancy"] * result["n_agents"]
    result["efficiency_x_tools"] = result["efficiency"] * result["n_tools"]
    result["baseline_x_agents"] = result["single_agent_baseline"] * result["log_agents"]
    result["error_x_tools"] = result["error_amplification"] * result["n_tools"]
    return result


def add_capability_features(
    frame: pd.DataFrame,
    center: float,
    intelligence: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    raw = result["model"].map(intelligence)
    if raw.isna().any():
        missing = sorted(result.loc[raw.isna(), "model"].unique())
        raise ValueError(f"Missing v4.1 Intelligence Index scores: {missing}")

    result["capability_centered"] = raw - center
    result["capability_sq_centered"] = result["capability_centered"] ** 2
    result["capability_x_efficiency"] = result["capability_centered"] * result["efficiency"]
    result["message_density_x_capability"] = result["message_density"] * result["capability_centered"]
    result["capability_x_tools"] = result["capability_centered"] * result["log_tools"]
    return result


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0.0:
        raise ValueError("R-squared is undefined because the target has zero variance")
    return 1.0 - residual / total


def fit_ols(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def predict_ols(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    return design @ coefficients


def five_fold_r2(features: np.ndarray, target: np.ndarray) -> float:
    indices = np.arange(len(features))
    np.random.RandomState(42).shuffle(indices)
    folds = np.array_split(indices, 5)
    scores = []
    for test_indices in folds:
        train_indices = np.setdiff1d(indices, test_indices, assume_unique=True)
        coefficients = fit_ols(features[train_indices], target[train_indices])
        prediction = predict_ols(features[test_indices], coefficients)
        scores.append(r_squared(target[test_indices], prediction))
    return float(np.mean(scores))


def evaluate_intelligence(
    training: pd.DataFrame,
    test: pd.DataFrame,
    intelligence: dict[str, float],
) -> dict[str, object]:
    training_shared = add_shared_features(training)
    test_shared = add_shared_features(test)
    training_capability = training_shared["model"].map(intelligence)
    if training_capability.isna().any():
        missing = sorted(training_shared.loc[training_capability.isna(), "model"].unique())
        raise ValueError(f"Missing v4.1 Intelligence Index scores: {missing}")
    center = float(training_capability.mean())

    training_features = add_capability_features(training_shared, center, intelligence)
    test_features = add_capability_features(test_shared, center, intelligence)

    means = training_features[FEATURES].mean()
    scales = training_features[FEATURES].std(ddof=0)
    if (scales == 0.0).any():
        constant = scales.index[scales == 0.0].tolist()
        raise ValueError(f"Constant prediction features: {constant}")
    x_train = ((training_features[FEATURES] - means) / scales).to_numpy(dtype=float)
    x_test = ((test_features[FEATURES] - means) / scales).to_numpy(dtype=float)
    y_train = training_features["performance"].to_numpy(dtype=float)
    y_test = test_features["performance"].to_numpy(dtype=float)

    coefficients = fit_ols(x_train, y_train)
    return {
        "capability": "Latest Intelligence Index v4.1",
        "source": "Our retraining",
        "n_train": len(training_features),
        "n_test": len(test_features),
        "train_r2": r_squared(y_train, predict_ols(x_train, coefficients)),
        "cv_folds": 5,
        "cv_r2": five_fold_r2(x_train, y_train),
        "test_r2": r_squared(y_test, predict_ols(x_test, coefficients)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    args = parser.parse_args()

    raw = args.data_dir / "raw"
    training = load_training(raw / "prediction_training.csv")
    test = load_test_runs(raw / "homogeneous_runs.csv")
    intelligence = load_intelligence(raw / "model_intelligence.csv")

    refitted = evaluate_intelligence(training, test, intelligence)
    results = pd.DataFrame(
        [
            {
                "capability": "Original Intelligence Index",
                "source": "Kim et al.",
                "n_train": 260,
                "n_test": None,
                "train_r2": 0.463,
                "cv_folds": 5,
                "cv_r2": 0.373,
                "test_r2": None,
            },
            refitted,
        ]
    )
    for column in ("n_train", "n_test", "cv_folds"):
        results[column] = results[column].astype("Int64")
    for column in ("train_r2", "cv_r2", "test_r2"):
        results[column] = results[column].round(3)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False, float_format="%.3f")

    display = results[["capability", "train_r2", "cv_r2", "test_r2"]].copy()
    for column in ["train_r2", "cv_r2", "test_r2"]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
