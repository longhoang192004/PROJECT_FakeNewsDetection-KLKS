# -*- coding: utf-8 -*-
"""
Leak-safe data splitting using StratifiedGroupKFold.

Ensures near-duplicate groups never span across train/val/test,
preventing data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple

from .dedup import report_leak_exact, report_leak_near


def leak_safe_split(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test with group-aware stratification.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'text_clean', 'label', 'group'.
    config : dict
        Must contain: 'test_ratio', 'val_ratio_of_trainval', 'seed',
        'near_dup_threshold', 'char_ngram_range', 'min_df'.

    Returns
    -------
    train_df, val_df, test_df : DataFrames (reset index)
    """
    seed = int(config["seed"])
    test_ratio = float(config["test_ratio"])
    val_ratio = float(config["val_ratio_of_trainval"])

    y = df["label"].values
    g = df["group"].values

    # ── Step 1: Split out test (≈10%) ───────────────────────────
    kfold_test = 10
    sgkf = StratifiedGroupKFold(
        n_splits=kfold_test, shuffle=True, random_state=seed
    )

    best_fold = None
    best_diff = 1e9
    for fold_i, (trainval_idx, test_idx) in enumerate(
        sgkf.split(df, y, groups=g)
    ):
        ratio = len(test_idx) / len(df)
        diff = abs(ratio - test_ratio)
        if diff < best_diff:
            best_diff = diff
            best_fold = (trainval_idx, test_idx, ratio, fold_i)

    trainval_idx, test_idx, ratio, fold_i = best_fold
    print(f"  Picked fold {fold_i} for TEST: "
          f"size={len(test_idx)} ({ratio:.3f})")

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ── Step 2: Split trainval → train + val (≈11% of trainval) ─
    kfold_val = 9
    sgkf2 = StratifiedGroupKFold(
        n_splits=kfold_val, shuffle=True, random_state=seed + 7
    )

    best2 = None
    best2_diff = 1e9
    for fold_i2, (train_idx, val_idx) in enumerate(
        sgkf2.split(
            df_trainval,
            df_trainval["label"].values,
            groups=df_trainval["group"].values,
        )
    ):
        ratio2 = len(val_idx) / len(df_trainval)
        diff2 = abs(ratio2 - val_ratio)
        if diff2 < best2_diff:
            best2_diff = diff2
            best2 = (train_idx, val_idx, ratio2, fold_i2)

    train_idx, val_idx, ratio2, fold_i2 = best2
    print(f"  Picked fold {fold_i2} for VAL: "
          f"size={len(val_idx)} ({ratio2:.3f})")

    train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
    val_df = df_trainval.iloc[val_idx].reset_index(drop=True)
    test_df = df_test.copy()

    # ── Print distribution ──────────────────────────────────────
    for name, dfx in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = len(dfx)
        c0 = int((dfx["label"] == 0).sum())
        c1 = int((dfx["label"] == 1).sum())
        ng = dfx["group"].nunique()
        print(f"  {name:5s}: {n:>5d} | "
              f"REAL={c0} ({c0/n:.1%}) | "
              f"FAKE={c1} ({c1/n:.1%}) | "
              f"groups={ng}")

    # ── Verify 0 group overlap ──────────────────────────────────
    overlap_tv = set(train_df["group"]) & set(val_df["group"])
    overlap_tt = set(train_df["group"]) & set(test_df["group"])
    overlap_vt = set(val_df["group"]) & set(test_df["group"])
    print(f"\n  Group overlap: "
          f"Train&Val={len(overlap_tv)} | "
          f"Train&Test={len(overlap_tt)} | "
          f"Val&Test={len(overlap_vt)}")

    if len(overlap_tv) > 0 or len(overlap_tt) > 0 or len(overlap_vt) > 0:
        print("  [!] WARNING: Group overlap detected! Split may have leaks.")

    return train_df, val_df, test_df


def run_leak_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Run full exact + near-dup leak report between all split pairs."""
    threshold = float(config["near_dup_threshold"])
    ngram_range = tuple(config["char_ngram_range"])
    min_df = int(config["min_df"])

    print("\n  -- Exact Leak Check --")
    exact_tv = report_leak_exact(
        train_df["text_clean"].tolist(),
        val_df["text_clean"].tolist(),
        "Train&Val",
    )
    exact_tt = report_leak_exact(
        train_df["text_clean"].tolist(),
        test_df["text_clean"].tolist(),
        "Train&Test",
    )
    exact_vt = report_leak_exact(
        val_df["text_clean"].tolist(),
        test_df["text_clean"].tolist(),
        "Val&Test",
    )

    print("\n  -- Near-dup Leak Check --")
    near_tv = report_leak_near(
        train_df["text_clean"].tolist(),
        val_df["text_clean"].tolist(),
        threshold=threshold,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    near_tt = report_leak_near(
        train_df["text_clean"].tolist(),
        test_df["text_clean"].tolist(),
        threshold=threshold,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    near_vt = report_leak_near(
        val_df["text_clean"].tolist(),
        test_df["text_clean"].tolist(),
        threshold=threshold,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    print(f"  Val->Train:  {near_tv}")
    print(f"  Test->Train: {near_tt}")
    print(f"  Test->Val:   {near_vt}")

    return {
        "exact": {
            "train_val": exact_tv,
            "train_test": exact_tt,
            "val_test": exact_vt,
        },
        "near_dup": {
            "val_to_train": near_tv,
            "test_to_train": near_tt,
            "test_to_val": near_vt,
        },
    }
