# -*- coding: utf-8 -*-
"""
PhoBERT Fine-tuning for Vietnamese Fake News Detection.

Uses PipelineV2 for data processing (clean, dedup, leak-safe split,
dynamic padding, text augmentation).

Usage:
    cd CodeKLKS
    python -m PipelineV2.train_phobert
    python -m PipelineV2.train_phobert --epochs 10 --batch_size 16
    python -m PipelineV2.train_phobert --data_path ../fakenewsdatasetv1.csv
"""

import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import get_config
from .pipeline import run_pipeline
from .dataset import (
    NewsDataset,
    single_collate_fn,
    compute_class_weights,
)
from torch.utils.data import DataLoader


# ============================================================
# TRAINING CONFIG
# ============================================================

TRAIN_CONFIG = {
    # -- model --
    "model_name": "vinai/phobert-base",
    "max_length": 256,

    # -- training --
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "max_grad_norm": 1.0,

    # -- freezing --
    "freeze_layers": 0,
    "freeze_embeddings": False,

    # -- augmentation (inherits from pipeline config) --
    "augment_prob": 0.3,
    "aug_delete_prob": 0.10,
    "aug_swap_prob": 0.10,

    # -- output --
    "output_dir": None,  # auto-set based on pipeline output_dir
}


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model, freeze_layers: int, freeze_embeddings: bool):
    """Freeze PhoBERT layers to reduce training time and prevent overfitting."""
    if freeze_layers <= 0 and not freeze_embeddings:
        return

    # PhoBERT uses RoBERTa architecture
    base = getattr(model, "roberta", None)
    if base is None:
        base = getattr(model, "bert", None)
    if base is None:
        print("  [!] Cannot find base model to freeze")
        return

    if freeze_embeddings and hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
        print(f"  Froze embeddings")

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        layers = base.encoder.layer
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        if freeze_layers > 0:
            print(f"  Froze first {freeze_layers}/{len(layers)} encoder layers")


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer_probs(model, dataloader, device):
    """Run inference, return P(FAKE) and confidence for each sample."""
    model.eval()
    p_fake_all, conf_all = [], []

    for batch in tqdm(dataloader, desc="Infer", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out.logits, dim=1)

        p_fake = probs[:, 1].cpu().numpy()
        conf = np.maximum(p_fake, 1.0 - p_fake)

        p_fake_all.extend(p_fake.tolist())
        conf_all.extend(conf.tolist())

    return (
        np.array(p_fake_all, dtype=np.float32),
        np.array(conf_all, dtype=np.float32),
    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(y_true, p_fake, thr_min=0.30, thr_max=0.70, step=0.01):
    """Search for threshold that maximizes macro F1."""
    best_thr, best_f1 = 0.5, -1.0
    thr = thr_min
    while thr <= thr_max + 1e-12:
        pred = (p_fake >= thr).astype(int)
        f1m = f1_score(y_true, pred, average="macro")
        if f1m > best_f1:
            best_f1 = float(f1m)
            best_thr = float(thr)
        thr += step
    return best_thr, best_f1


# ============================================================
# TRAINING LOOP
# ============================================================

def train_phobert(pipeline_config: dict, train_config: dict):
    """
    Full PhoBERT training pipeline:
      1. Run PipelineV2 (clean, dedup, split)
      2. Create DataLoaders with dynamic padding + augmentation
      3. Fine-tune PhoBERT
      4. Evaluate on test set
      5. Save model + results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(pipeline_config.get("seed", 42))
    set_seed(seed)

    print("=" * 80)
    print("PHOBERT FINE-TUNING FOR FAKE NEWS DETECTION")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Model:  {train_config['model_name']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch:  {train_config['batch_size']}")
    print(f"  LR:     {train_config['learning_rate']}")
    print()

    # ── 1. Run data pipeline ────────────────────────────────────
    splits = run_pipeline(config=pipeline_config, verbose=True)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # ── 2. Load tokenizer ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    model_name = train_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"  Tokenizer: {model_name} (vocab={len(tokenizer)})")

    # ── 3. Create DataLoaders ───────────────────────────────────
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS (DYNAMIC PADDING + AUGMENTATION)")
    print("=" * 80)

    max_length = int(train_config["max_length"])
    batch_size = int(train_config["batch_size"])
    aug_prob = float(train_config.get("augment_prob", 0.0))

    train_dataset = NewsDataset(
        texts=train_df["text_clean"].values,
        labels=train_df["label"].values,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=True,
        augment_prob=aug_prob,
        aug_delete_prob=float(train_config.get("aug_delete_prob", 0.10)),
        aug_swap_prob=float(train_config.get("aug_swap_prob", 0.10)),
    )
    val_dataset = NewsDataset(
        texts=val_df["text_clean"].values,
        labels=val_df["label"].values,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = NewsDataset(
        texts=test_df["text_clean"].values,
        labels=test_df["label"].values,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=single_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=single_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=single_collate_fn,
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Augmentation: prob={aug_prob}")

    # ── 4. Load model ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # Set dropout
    dropout_val = float(train_config.get("dropout", 0.1))
    try:
        model.config.hidden_dropout_prob = dropout_val
        model.config.attention_probs_dropout_prob = dropout_val
    except Exception:
        pass

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    freeze_layers = int(train_config.get("freeze_layers", 0))
    freeze_emb = bool(train_config.get("freeze_embeddings", False))
    freeze_backbone(model, freeze_layers, freeze_emb)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,} ({trainable/total_params:.1%})")

    # ── 5. Loss, optimizer, scheduler ───────────────────────────
    y_train = train_df["label"].values.astype(int)
    counts, class_w = compute_class_weights(y_train, device=device)
    print(f"  Class counts [REAL, FAKE]: {counts}")
    print(f"  Class weights: {class_w.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    epochs = int(train_config["epochs"])
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * float(train_config.get("warmup_ratio", 0.1)))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # ── 6. Training loop ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(out.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_config.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # -- Validation --
        avg_loss = total_loss / max(n_batches, 1)
        p_val, c_val = infer_probs(model, val_loader, device)
        y_val = val_df["label"].values.astype(int)

        pred_val = (p_val >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, pred_val)
        val_f1 = f1_score(y_val, pred_val, average="macro")

        epoch_info = {
            "epoch": ep + 1,
            "train_loss": round(avg_loss, 4),
            "val_acc": round(float(val_acc), 4),
            "val_f1_macro": round(float(val_f1), 4),
        }
        history.append(epoch_info)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            marker = " << BEST"

        print(
            f"  Epoch {ep+1}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_acc={val_acc*100:.2f}% | "
            f"val_F1={val_f1:.4f}{marker}"
        )

    # ── 7. Load best & evaluate ─────────────────────────────────
    print("\n" + "=" * 80)
    print(f"EVALUATION (best epoch = {best_epoch})")
    print("=" * 80)

    if best_state is not None:
        model.load_state_dict(best_state)

    # -- Optimal threshold on val --
    p_val, _ = infer_probs(model, val_loader, device)
    y_val = val_df["label"].values.astype(int)
    best_thr, best_val_f1 = find_best_threshold(y_val, p_val)
    print(f"  Optimal threshold (val): {best_thr:.2f} (F1={best_val_f1:.4f})")

    # -- Test evaluation --
    p_test, c_test = infer_probs(model, test_loader, device)
    y_test = test_df["label"].values.astype(int)
    pred_test = (p_test >= best_thr).astype(int)

    test_acc = accuracy_score(y_test, pred_test)
    test_f1_macro = f1_score(y_test, pred_test, average="macro")
    test_f1_fake = f1_score(y_test, pred_test, pos_label=1, average="binary")
    test_f1_real = f1_score(y_test, pred_test, pos_label=0, average="binary")

    print(f"\n  TEST RESULTS:")
    print(f"  Accuracy:     {test_acc*100:.2f}%")
    print(f"  Macro F1:     {test_f1_macro:.4f}")
    print(f"  F1 (FAKE):    {test_f1_fake:.4f}")
    print(f"  F1 (REAL):    {test_f1_real:.4f}")
    print(f"  Gap:          {abs(test_f1_fake - test_f1_real):.4f}")
    print(f"  Threshold:    {best_thr:.2f}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, pred_test,
        target_names=["REAL (0)", "FAKE (1)"],
        digits=4,
    ))

    # ── 8. Save model + results ─────────────────────────────────
    output_dir = train_config.get("output_dir")
    if output_dir is None:
        output_dir = Path(pipeline_config["output_dir"]) / "phobert_model"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = output_dir / "best_model_state_dict.pt"
    torch.save(best_state or model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    print(f"  Tokenizer saved: {output_dir / 'tokenizer'}")

    # Save results
    results = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_threshold": best_thr,
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(test_f1_macro), 4),
        "test_f1_fake": round(float(test_f1_fake), 4),
        "test_f1_real": round(float(test_f1_real), 4),
        "training_history": history,
        "train_config": {
            k: v for k, v in train_config.items()
            if isinstance(v, (str, int, float, bool, list))
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved: {output_dir / 'results.json'}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE [OK]")
    print("=" * 80)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "results": results,
        "best_threshold": best_thr,
        "p_test": p_test,
        "c_test": c_test,
    }


# ============================================================
# QUICK PREDICTION (for testing after training)
# ============================================================

def predict_text(text: str, model, tokenizer, threshold=0.5, device=None):
    """Predict a single text input."""
    from .clean_text import clean_text

    if device is None:
        device = next(model.parameters()).device

    text_clean = clean_text(text)
    enc = tokenizer(
        text_clean,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out.logits, dim=1)
        p_fake = float(probs[0, 1].cpu().item())

    label = "FAKE" if p_fake >= threshold else "REAL"
    confidence = max(p_fake, 1.0 - p_fake)

    return {
        "label": label,
        "p_fake": round(p_fake, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PhoBERT Fine-tuning for Fake News Detection (PipelineV2)"
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None,
                        help="Directory to save trained model (default: <output_dir>/phobert_model)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--augment_prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Pipeline config
    pipeline_overrides = {"seed": args.seed}
    if args.data_path:
        pipeline_overrides["data_path"] = str(Path(args.data_path).resolve())
    if args.output_dir:
        pipeline_overrides["output_dir"] = str(Path(args.output_dir).resolve())

    pipeline_cfg = get_config(**pipeline_overrides)

    # Training config
    train_cfg = TRAIN_CONFIG.copy()
    train_cfg["epochs"] = args.epochs
    train_cfg["batch_size"] = args.batch_size
    train_cfg["learning_rate"] = args.learning_rate
    train_cfg["max_length"] = args.max_length
    train_cfg["freeze_layers"] = args.freeze_layers
    train_cfg["freeze_embeddings"] = args.freeze_embeddings
    train_cfg["augment_prob"] = args.augment_prob
    if args.model_output_dir:
        train_cfg["output_dir"] = str(Path(args.model_output_dir).resolve())

    train_phobert(pipeline_cfg, train_cfg)


if __name__ == "__main__":
    main()
