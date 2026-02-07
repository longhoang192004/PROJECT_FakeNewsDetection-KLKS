# infer_cross_attn.py
# ============================================================
# INFERENCE ONLY (NO TRAIN): Cross-Attention Fusion
# - Load trained model: best_model_state_dict.pt
# - Load calibration: calibration_temperature.json + calibration_threshold.json
# - CLI: paste text -> get REAL/FAKE + p_fake + conf
# ============================================================

import os
import re
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import emoji


# ============================================================
# CONFIG (CHỈ SỬA MẤY DÒNG NÀY)
# ============================================================

ELECTRA_NAME = "FPTAI/velectra-base-discriminator-cased"
PHOBERT_NAME = "vinai/phobert-base"

# Folder output đã train xong (có best_model_state_dict.pt + calibration_*.json)
RUN_DIR = Path(r"D:\KLKS\PROJECT_FakeNewsDetection-KLKS\outputs\cross_attn_fusion_20260207_211344")

MODEL_PATH = RUN_DIR / "best_model_state_dict.pt"
TEMP_PATH  = RUN_DIR / "calibration_temperature.json"
THR_PATH   = RUN_DIR / "calibration_threshold.json"

MAX_LEN_ELECTRA = 256
MAX_LEN_PHOBERT = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TEXT CLEANING (GIỐNG FILE TRAIN)
# ============================================================

def emoji_to_text(text: str) -> str:
    if not text:
        return ""
    t = emoji.demojize(text, delimiters=(" ", " "))
    t = t.replace("_", " ")
    t = t.replace("\uFE0F", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = emoji_to_text(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid(text):
    return len(text.split()) >= 8


# ============================================================
# MODEL (GIỐNG FILE TRAIN)
# ============================================================

class CrossAttnFusionClassifier(nn.Module):
    def __init__(self, electra_name, phobert_name, dropout=0.3, heads=8, cross_attn_dropout=0.1):
        super().__init__()
        self.e = AutoModel.from_pretrained(electra_name)
        self.p = AutoModel.from_pretrained(phobert_name)

        He = self.e.config.hidden_size
        Hp = self.p.config.hidden_size

        self.proj_kv = nn.Linear(He, Hp) if He != Hp else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=Hp,
            num_heads=heads,
            dropout=cross_attn_dropout,
            batch_first=False
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(Hp)

        self.head = nn.Sequential(
            nn.Linear(Hp * 2, Hp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(Hp, 2)
        )

    def forward(self, e_input_ids, e_attn, p_input_ids, p_attn):
        e_out = self.e(input_ids=e_input_ids, attention_mask=e_attn, return_dict=True)
        p_out = self.p(input_ids=p_input_ids, attention_mask=p_attn, return_dict=True)

        E = e_out.last_hidden_state
        P = p_out.last_hidden_state

        E2 = self.proj_kv(E)

        Q = P.transpose(0, 1)
        K = E2.transpose(0, 1)
        V = E2.transpose(0, 1)

        e_key_pad = (e_attn == 0)

        cross, _ = self.cross_attn(Q, K, V, key_padding_mask=e_key_pad)
        cross = cross.transpose(0, 1)

        cross = self.norm(cross + P)

        cls_p = P[:, 0, :]

        mask = p_attn.unsqueeze(-1).float()
        cross_sum = (cross * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        cross_mean = cross_sum / denom

        feat = torch.cat([cls_p, cross_mean], dim=1)
        feat = self.dropout(feat)
        logits = self.head(feat)
        return logits


# ============================================================
# LOAD TOKENIZERS + LOAD MODEL WEIGHTS + LOAD CALIBRATION
# ============================================================

def load_temperature(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("temperature", None))

def load_threshold(path: Path):
    if not path.exists():
        return 0.5
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("threshold", 0.5))

def assert_paths():
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"RUN_DIR not found: {RUN_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    # TEMP/THR có thể không có nếu bạn tắt calibration, nên không raise ở đây

def build_model_and_load():
    print("=" * 90)
    print("INFERENCE ONLY: CROSS-ATTENTION FUSION")
    print("=" * 90)
    print("Device:", DEVICE)
    print("RUN_DIR:", RUN_DIR)
    print("MODEL_PATH:", MODEL_PATH)

    print("\nLoading tokenizers...")
    tokE = AutoTokenizer.from_pretrained(ELECTRA_NAME, use_fast=False)
    tokP = AutoTokenizer.from_pretrained(PHOBERT_NAME, use_fast=False)

    print("Building model...")
    model = CrossAttnFusionClassifier(ELECTRA_NAME, PHOBERT_NAME).to(DEVICE)

    print("Loading state_dict...")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    T = load_temperature(TEMP_PATH)
    thr = load_threshold(THR_PATH)

    print("\nCalibration loaded:")
    print(" - Temperature (T):", T)
    print(" - Threshold (thr):", thr)
    print("=" * 90)

    return model, tokE, tokP, T, thr


# ============================================================
# PREDICT ONE
# ============================================================

@torch.no_grad()
def predict_one(model, tokE, tokP, text: str, T=None, thr=0.5):
    text_clean = clean_text(text)
    if not is_valid(text_clean):
        return {"ok": False, "error": "Text too short after cleaning", "text_clean": text_clean}

    encE = tokE(
        text_clean,
        max_length=MAX_LEN_ELECTRA,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encP = tokP(
        text_clean,
        max_length=MAX_LEN_PHOBERT,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    e_input_ids = encE["input_ids"].to(DEVICE)
    e_attn      = encE["attention_mask"].to(DEVICE)
    p_input_ids = encP["input_ids"].to(DEVICE)
    p_attn      = encP["attention_mask"].to(DEVICE)

    logits = model(e_input_ids, e_attn, p_input_ids, p_attn)

    # Apply temperature scaling if available
    if T is not None:
        Tt = torch.tensor(float(T), device=logits.device, dtype=logits.dtype).clamp(min=1e-3)
        logits = logits / Tt

    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    p_fake = float(probs[1])

    pred = 1 if p_fake >= float(thr) else 0
    conf = max(p_fake, 1.0 - p_fake)

    return {
        "ok": True,
        "pred": pred,              # 0=REAL, 1=FAKE
        "p_fake": p_fake,
        "conf": conf,
        "thr": float(thr),
        "T": (None if T is None else float(T)),
        "text_clean": text_clean
    }


# ============================================================
# CLI LOOP
# ============================================================

def cli_loop(model, tokE, tokP, T, thr):
    print("\nNhập bài báo để phân loại (gõ 'exit' để thoát).")
    print("Tip: dán nguyên đoạn dài cũng được.\n")

    while True:
        s = input("> ").strip()
        if s.lower() in ["exit", "quit", "q"]:
            break

        out = predict_one(model, tokE, tokP, s, T=T, thr=thr)
        if not out["ok"]:
            print("❌", out["error"])
            continue

        label = "FAKE (1)" if out["pred"] == 1 else "REAL (0)"
        print(f"✅ Pred: {label} | p_fake={out['p_fake']:.3f} | conf={out['conf']:.3f} | thr={out['thr']:.2f} | T={out['T']}")


def main():
    assert_paths()
    model, tokE, tokP, T, thr = build_model_and_load()
    cli_loop(model, tokE, tokP, T, thr)


if __name__ == "__main__":
    main()
