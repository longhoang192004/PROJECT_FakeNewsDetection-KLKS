# -*- coding: utf-8 -*-
"""
Dataset and DataLoader utilities with:
  - Dynamic padding (pad to max-in-batch, not global max)
  - Text augmentation (random word delete / swap) for training
  - Support for single-tokenizer and dual-tokenizer modes
"""

import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Text Augmentation ──────────────────────────────────────────

def augment_text(
    text: str,
    delete_prob: float = 0.10,
    swap_prob: float = 0.10,
) -> str:
    """
    Simple text augmentation via random word deletion and word swap.

    Parameters
    ----------
    text : str
        Input text.
    delete_prob : float
        For each word, probability of deleting it.
    swap_prob : float
        Probability of performing one random adjacent-word swap.
    """
    words = text.split()
    if len(words) <= 3:
        return text  # too short to augment safely

    # Random word deletion (keep at least 50% of words)
    if delete_prob > 0:
        kept = [w for w in words if random.random() > delete_prob]
        if len(kept) >= max(3, len(words) // 2):
            words = kept

    # Random adjacent-word swap
    if swap_prob > 0 and random.random() < swap_prob and len(words) >= 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]

    return " ".join(words)


# ── Single-Tokenizer Dataset ──────────────────────────────────

class NewsDataset(Dataset):
    """
    Dataset for a single transformer tokenizer.
    Tokenizes lazily; padding happens in the collate function.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        is_train: bool = False,
        augment_prob: float = 0.0,
        aug_delete_prob: float = 0.10,
        aug_swap_prob: float = 0.10,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.aug_delete_prob = aug_delete_prob
        self.aug_swap_prob = aug_swap_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Apply augmentation during training
        if self.is_train and self.augment_prob > 0 and random.random() < self.augment_prob:
            text = augment_text(
                text,
                delete_prob=self.aug_delete_prob,
                swap_prob=self.aug_swap_prob,
            )

        # Tokenize WITHOUT padding — padding is done in collate_fn
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,           # << dynamic padding
            return_tensors=None,     # return plain lists
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": label,
        }


def single_collate_fn(batch):
    """
    Dynamic padding: pad all sequences in the batch to the
    longest sequence in that batch (not the global max_length).
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - len(ids)

        # Pad with 0 (standard pad token id)
        input_ids.append(ids + [0] * pad_len)
        attention_masks.append(mask + [0] * pad_len)
        labels.append(item["label"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


# ── Dual-Tokenizer Dataset ────────────────────────────────────

class DualTokDataset(Dataset):
    """
    Dataset for two tokenizers (e.g., ELECTRA + PhoBERT).
    Each sample is tokenized by both; padding in collate.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tok_electra,
        tok_phobert,
        max_electra: int = 256,
        max_phobert: int = 256,
        is_train: bool = False,
        augment_prob: float = 0.0,
        aug_delete_prob: float = 0.10,
        aug_swap_prob: float = 0.10,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok_e = tok_electra
        self.tok_p = tok_phobert
        self.max_e = int(max_electra)
        self.max_p = int(max_phobert)
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.aug_delete_prob = aug_delete_prob
        self.aug_swap_prob = aug_swap_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        if self.is_train and self.augment_prob > 0 and random.random() < self.augment_prob:
            text = augment_text(
                text,
                delete_prob=self.aug_delete_prob,
                swap_prob=self.aug_swap_prob,
            )

        enc_e = self.tok_e(
            text, max_length=self.max_e, truncation=True,
            padding=False, return_tensors=None,
        )
        enc_p = self.tok_p(
            text, max_length=self.max_p, truncation=True,
            padding=False, return_tensors=None,
        )

        return {
            "e_input_ids": enc_e["input_ids"],
            "e_attn": enc_e["attention_mask"],
            "p_input_ids": enc_p["input_ids"],
            "p_attn": enc_p["attention_mask"],
            "label": label,
        }


def dual_collate_fn(batch):
    """Dynamic padding for dual-tokenizer batch."""
    max_e = max(len(item["e_input_ids"]) for item in batch)
    max_p = max(len(item["p_input_ids"]) for item in batch)

    e_ids, e_attn = [], []
    p_ids, p_attn = [], []
    labels = []

    for item in batch:
        # ELECTRA padding
        pad_e = max_e - len(item["e_input_ids"])
        e_ids.append(item["e_input_ids"] + [0] * pad_e)
        e_attn.append(item["e_attn"] + [0] * pad_e)

        # PhoBERT padding
        pad_p = max_p - len(item["p_input_ids"])
        p_ids.append(item["p_input_ids"] + [1] * pad_p)  # PhoBERT pad_token_id = 1
        p_attn.append(item["p_attn"] + [0] * pad_p)

        labels.append(item["label"])

    return {
        "e_input_ids": torch.tensor(e_ids, dtype=torch.long),
        "e_attn": torch.tensor(e_attn, dtype=torch.long),
        "p_input_ids": torch.tensor(p_ids, dtype=torch.long),
        "p_attn": torch.tensor(p_attn, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


# ── DataLoader Factories ──────────────────────────────────────

def make_single_loaders(
    train_df,
    val_df,
    test_df,
    tokenizer,
    config: dict,
):
    """
    Create DataLoaders for a single-tokenizer model.
    Train loader has augmentation + shuffle; val/test do not.
    """
    max_length = int(config.get("max_length_electra", 256))
    batch_size = int(config.get("batch_size", 16))
    aug_prob = float(config.get("augment_prob", 0.0))
    del_prob = float(config.get("aug_delete_prob", 0.10))
    swap_prob = float(config.get("aug_swap_prob", 0.10))

    train_loader = DataLoader(
        NewsDataset(
            train_df["text_clean"].values,
            train_df["label"].values,
            tokenizer, max_length,
            is_train=True,
            augment_prob=aug_prob,
            aug_delete_prob=del_prob,
            aug_swap_prob=swap_prob,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=single_collate_fn,
    )
    val_loader = DataLoader(
        NewsDataset(
            val_df["text_clean"].values,
            val_df["label"].values,
            tokenizer, max_length,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=single_collate_fn,
    )
    test_loader = DataLoader(
        NewsDataset(
            test_df["text_clean"].values,
            test_df["label"].values,
            tokenizer, max_length,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=single_collate_fn,
    )
    return train_loader, val_loader, test_loader


def make_dual_loaders(
    train_df,
    val_df,
    test_df,
    tok_electra,
    tok_phobert,
    config: dict,
):
    """
    Create DataLoaders for dual-tokenizer models (CrossAttention, Gating).
    Train loader has augmentation + shuffle; val/test do not.
    """
    max_e = int(config.get("max_length_electra", 256))
    max_p = int(config.get("max_length_phobert", 256))
    batch_size = int(config.get("batch_size", 4))
    aug_prob = float(config.get("augment_prob", 0.0))
    del_prob = float(config.get("aug_delete_prob", 0.10))
    swap_prob = float(config.get("aug_swap_prob", 0.10))

    train_loader = DataLoader(
        DualTokDataset(
            train_df["text_clean"].values,
            train_df["label"].values,
            tok_electra, tok_phobert,
            max_e, max_p,
            is_train=True,
            augment_prob=aug_prob,
            aug_delete_prob=del_prob,
            aug_swap_prob=swap_prob,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dual_collate_fn,
    )
    val_loader = DataLoader(
        DualTokDataset(
            val_df["text_clean"].values,
            val_df["label"].values,
            tok_electra, tok_phobert,
            max_e, max_p,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dual_collate_fn,
    )
    test_loader = DataLoader(
        DualTokDataset(
            test_df["text_clean"].values,
            test_df["label"].values,
            tok_electra, tok_phobert,
            max_e, max_p,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dual_collate_fn,
    )
    return train_loader, val_loader, test_loader


# ── Class Weights ──────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, device=None):
    """
    Compute inverse-frequency class weights for imbalanced data.

    Returns
    -------
    counts : np.ndarray [n_classes]
    weights : torch.Tensor [n_classes]
    """
    counts = np.bincount(labels.astype(int), minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        w_tensor = w_tensor.to(device)
    return counts, w_tensor
