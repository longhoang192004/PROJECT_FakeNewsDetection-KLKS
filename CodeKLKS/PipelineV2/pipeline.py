# -*- coding: utf-8 -*-
"""
Pipeline orchestrator.

Runs the full data processing pipeline:
  Load CSV → Clean → Filter → Dedup → Group → Split → Leak Report → Save
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .clean_text import clean_text, is_valid
from .dedup import build_near_dup_groups
from .split import leak_safe_split, run_leak_report
from .config import get_config


def run_pipeline(
    config: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full data processing pipeline.

    Parameters
    ----------
    config : dict, optional
        Configuration dict. If None, uses default CONFIG.
    verbose : bool
        Print progress info.

    Returns
    -------
    dict with keys 'train', 'val', 'test' → DataFrames
    """
    if config is None:
        config = get_config()

    data_path = config["data_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CSV ─────────────────────────────────────────────
    if verbose:
        print("=" * 80)
        print("PIPELINE V2 - LOAD DATA")
        print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Auto-detect columns
    if "text" not in df.columns:
        for c in ["content", "article", "news", "body", "title", "NỘI DUNG"]:
            if c in df.columns:
                df["text"] = df[c]
                break
    if "text" not in df.columns:
        raise ValueError(
            f"Cannot find text column. Available: {list(df.columns)}"
        )

    if "label" not in df.columns:
        for c in ["class", "category", "y", "GIẢ(0)/THẬT(1)"]:
            if c in df.columns:
                df["label"] = df[c]
                break
    if "label" not in df.columns:
        raise ValueError(
            f"Cannot find label column. Available: {list(df.columns)}"
        )

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    bad = df[~df["label"].isin([0, 1])]
    if len(bad) > 0:
        raise ValueError(
            f"Found labels not in {{0, 1}}. Examples:\n{bad.head()}"
        )

    initial_count = len(df)
    if verbose:
        print(f"  Loaded: {initial_count} samples")

    # ── 2. Clean text ───────────────────────────────────────────
    if verbose:
        print("\n  Cleaning text (unified pipeline)...")

    df["text_clean"] = df["text"].apply(clean_text)

    # ── 3. Filter invalid ───────────────────────────────────────
    min_words = int(config.get("min_words", 8))
    min_chars = int(config.get("min_chars", 10))
    df = df[
        df["text_clean"].apply(lambda t: is_valid(t, min_words, min_chars))
    ].copy()

    after_filter = len(df)
    if verbose:
        print(f"  After filter (>={min_words} words, >={min_chars} chars): "
              f"{after_filter} (removed {initial_count - after_filter})")

    # ── 4. Exact dedup ──────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text_clean"], keep="first")
    df = df.reset_index(drop=True)

    after_dedup = len(df)
    if verbose:
        print(f"  After exact dedup: {after_dedup} "
              f"(removed {before_dedup - after_dedup})")

    n = len(df)
    c0 = int((df["label"] == 0).sum())
    c1 = int((df["label"] == 1).sum())
    if verbose:
        print(f"\n  Dataset: {n} samples | "
              f"REAL={c0} ({c0/n:.1%}) | FAKE={c1} ({c1/n:.1%})")

    # ── 5. Near-dup clustering ──────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("BUILDING NEAR-DUP CLUSTERS")
        print("=" * 80)

    groups = build_near_dup_groups(
        df["text_clean"].tolist(),
        threshold=float(config["near_dup_threshold"]),
        k=int(config["near_dup_k"]),
        ngram_range=tuple(config["char_ngram_range"]),
        min_df=int(config["min_df"]),
    )
    df["group"] = groups

    n_groups = int(df["group"].nunique())
    group_sizes = df["group"].value_counts()
    if verbose:
        print(f"  Groups: {n_groups} | "
              f"Largest group size: {int(group_sizes.max())}")
        print(f"  Top 5 group sizes:\n{group_sizes.head(5).to_string()}")

    # ── 6. Leak-safe split ──────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("LEAK-SAFE SPLIT (STRATIFIED + GROUP-AWARE)")
        print("=" * 80)

    train_df, val_df, test_df = leak_safe_split(df, config)

    # ── 7. Leak report ──────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("LEAK REPORT")
        print("=" * 80)

    leak_report = run_leak_report(train_df, val_df, test_df, config)

    # ── 8. Save outputs ────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)

    train_df.to_csv(
        output_dir / "train.csv", index=False, encoding="utf-8-sig"
    )
    val_df.to_csv(
        output_dir / "val.csv", index=False, encoding="utf-8-sig"
    )
    test_df.to_csv(
        output_dir / "test.csv", index=False, encoding="utf-8-sig"
    )

    # Save dataset with all groups (for reproducibility)
    df.to_csv(
        output_dir / "dataset_clean_dedup.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save leak report
    with open(output_dir / "leak_report.json", "w", encoding="utf-8") as f:
        json.dump(leak_report, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats = {
        "initial_count": initial_count,
        "after_filter": after_filter,
        "after_dedup": after_dedup,
        "n_groups": n_groups,
        "largest_group": int(group_sizes.max()),
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_distribution": {
            "total": {"REAL": c0, "FAKE": c1},
            "train": {
                "REAL": int((train_df["label"] == 0).sum()),
                "FAKE": int((train_df["label"] == 1).sum()),
            },
            "val": {
                "REAL": int((val_df["label"] == 0).sum()),
                "FAKE": int((val_df["label"] == 1).sum()),
            },
            "test": {
                "REAL": int((test_df["label"] == 0).sum()),
                "FAKE": int((test_df["label"] == 1).sum()),
            },
        },
        "config": {
            k: v for k, v in config.items()
            if isinstance(v, (str, int, float, bool, list, tuple))
        },
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  Saved to: {output_dir}")
        print(f"    - train.csv      ({len(train_df)} samples)")
        print(f"    - val.csv        ({len(val_df)} samples)")
        print(f"    - test.csv       ({len(test_df)} samples)")
        print(f"    - dataset_clean_dedup.csv")
        print(f"    - leak_report.json")
        print(f"    - stats.json")
        print("\n" + "=" * 80)
        print("PIPELINE V2 COMPLETE [OK]")
        print("=" * 80)

    return {"train": train_df, "val": val_df, "test": test_df}
