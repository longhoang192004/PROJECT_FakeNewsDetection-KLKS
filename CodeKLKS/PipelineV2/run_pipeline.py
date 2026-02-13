# -*- coding: utf-8 -*-
"""
CLI entrypoint for PipelineV2.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data_path ../../fakenewsdatasetv1.csv
    python run_pipeline.py --data_path ../../fakenewsdatasetv1.csv --output_dir ../../outputs/my_run
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script from the PipelineV2 directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from PipelineV2.config import get_config
from PipelineV2.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="PipelineV2 — Unified data processing for Fake News Detection"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to input CSV (default: fakenewsdatasetv1.csv in project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files (default: outputs/pipeline_v2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=8,
        help="Minimum word count to keep a sample (default: 8)",
    )
    parser.add_argument(
        "--near_dup_threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for near-dup detection (default: 0.92)",
    )

    args = parser.parse_args()

    # Build config with CLI overrides
    overrides = {}
    if args.data_path is not None:
        overrides["data_path"] = str(Path(args.data_path).resolve())
    if args.output_dir is not None:
        overrides["output_dir"] = str(Path(args.output_dir).resolve())
    if args.seed != 42:
        overrides["seed"] = args.seed
    if args.min_words != 8:
        overrides["min_words"] = args.min_words
    if args.near_dup_threshold != 0.92:
        overrides["near_dup_threshold"] = args.near_dup_threshold

    config = get_config(**overrides)

    print(f"\n[*] Data path:   {config['data_path']}")
    print(f"[*] Output dir:  {config['output_dir']}")
    print(f"[*] Seed:        {config['seed']}")
    print(f"[*] Min words:   {config['min_words']}")
    print(f"[*] Near-dup thr:{config['near_dup_threshold']}")
    print()

    result = run_pipeline(config=config, verbose=True)

    # Quick summary
    total = sum(len(result[k]) for k in ["train", "val", "test"])
    print(f"\n[OK] Total processed: {total} samples")
    for k in ["train", "val", "test"]:
        print(f"   {k:5s}: {len(result[k]):>5d} ({len(result[k])/total:.1%})")


if __name__ == "__main__":
    main()
