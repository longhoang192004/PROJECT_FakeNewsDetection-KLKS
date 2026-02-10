# -*- coding: utf-8 -*-
"""
Centralized configuration for PipelineV2.
All parameters in one place — no more per-script duplication.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # CodeKLKS/../ == PROJECT_FakeNewsDetection-KLKS

CONFIG = {
    # ── data paths ──────────────────────────────────────────────
    "data_path": str(PROJECT_ROOT / "fakenewsdatasetv1.csv"),
    "output_dir": str(PROJECT_ROOT / "outputs" / "pipeline_v2"),

    # ── text cleaning ───────────────────────────────────────────
    "min_words": 8,           # sentences shorter than this are discarded
    "min_chars": 10,          # fallback: discard texts shorter than 10 chars

    # ── near-dup detection ──────────────────────────────────────
    "near_dup_threshold": 0.92,
    "near_dup_k": 20,
    "char_ngram_range": (4, 6),
    "min_df": 2,

    # ── split ratios ────────────────────────────────────────────
    "test_ratio": 0.10,
    "val_ratio_of_trainval": 0.11,   # ≈ 10% of total

    # ── tokenizer / model names ─────────────────────────────────
    "electra_name": "FPTAI/velectra-base-discriminator-cased",
    "phobert_name": "vinai/phobert-base",
    "max_length_electra": 256,
    "max_length_phobert": 256,

    # ── augmentation (training only) ────────────────────────────
    "augment_prob": 0.3,       # probability of augmenting a sample
    "aug_delete_prob": 0.10,   # per-word deletion probability
    "aug_swap_prob": 0.10,     # probability of doing a random swap

    # ── reproducibility ─────────────────────────────────────────
    "seed": 42,
}


def get_config(**overrides) -> dict:
    """Return a copy of CONFIG with optional overrides."""
    cfg = CONFIG.copy()
    cfg.update(overrides)
    return cfg
