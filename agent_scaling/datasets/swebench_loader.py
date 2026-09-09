"""Download SWE-Bench Verified from HuggingFace and convert it to framework JSON.

Usage:
    python -m agent_scaling.datasets.swebench_loader --output datasets/swebench-verified.json
"""

import argparse
import json
import os

from datasets import load_dataset
from loguru import logger

CONFIG = {
    "hf_path": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "dataset_id": "swebench-verified",
    "expected_count": 500,
}

# Fields that may have inconsistent casing across HF datasets
FIELD_NORMALIZATION = {
    "FAIL_TO_PASS": "fail_to_pass",
    "PASS_TO_PASS": "pass_to_pass",
}


def normalize_instance(raw: dict) -> dict:
    """Normalize field names from HuggingFace dataset format."""
    normalized = {}
    for key, value in raw.items():
        norm_key = FIELD_NORMALIZATION.get(key, key)
        normalized[norm_key] = value
    return normalized


def load_and_convert(output_path: str) -> None:
    """Load dataset from HuggingFace and save as framework JSON."""
    config = CONFIG
    logger.info(f"Loading {config['hf_path']} split={config['split']}...")

    dataset = load_dataset(config["hf_path"], split=config["split"])
    logger.info(f"Loaded {len(dataset)} instances (expected: {config['expected_count']})")

    instances = []
    for row in dataset:
        instance = normalize_instance(dict(row))
        instances.append(instance)

    output = {
        "dataset_id": config["dataset_id"],
        "instances": instances,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(instances)} instances to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SWE-Bench dataset and convert to framework JSON format"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()
    load_and_convert(args.output)


if __name__ == "__main__":
    main()
