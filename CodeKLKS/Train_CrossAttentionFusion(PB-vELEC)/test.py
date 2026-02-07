# -*- coding: utf-8 -*-

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys, subprocess, warnings, time, json, re, random, copy
warnings.filterwarnings("ignore")
from pathlib import Path

def ensure_pkg(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

print("Installing packages (if missing)...")
for pkg in [
    ("transformers", "transformers"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("emoji", "emoji"),
]:
    ensure_pkg(pkg[0], pkg[1])

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import emoji

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"cross_attn_fusion_fixed_{RUN_ID}"

CONFIG = {
    # ---- MODE ----
    "mode": "train",   # "train" or "infer"
    "load_run_dir": r"",  # For infer mode

    # ---- data/output ----
    "data_path": str(PROJECT_ROOT / "fakenewsdatasetv1.csv"),
    "output_dir": str(DEFAULT_OUTPUT_DIR),

    # ---- backbones ----
    "electra_name": "FPTAI/velectra-base-discriminator-cased",
    "phobert_name": "vinai/phobert-base",

    # ---- token ----
    "max_length_electra": 256,
    "max_length_phobert": 256,

    # ---- train ----
    "batch_size": 4,
    "learning_rate": 2e-4,
    "epochs": 20,
    "warmup_steps": 80,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "grad_clip": 1.0,

    # ---- freeze strategy ----
    "freeze_backbones": True,
    "unfreeze_last_n_layers_electra": 2,
    "unfreeze_last_n_layers_phobert": 2,

    # ---- cross attn ----
    "cross_attn_heads": 8,
    "cross_attn_dropout": 0.1,

    # ---- leak-safe grouping ----
    "near_dup_threshold": 0.92,
    "near_dup_k": 20,
    "char_ngram_range": (4, 6),
    "min_df": 2,

    # ---- split ratios ----
    "test_ratio": 0.10,
    "val_ratio_of_trainval": 0.11,

    # ---- misc ----
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,

    # ---- losses ----
    "label_smoothing": 0.0,

    "use_focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,

    "use_confidence_penalty": True,
    "confidence_penalty_beta": 0.05,

    "use_mixup": True,
    "mixup_alpha": 0.2,
    "mixup_prob": 0.5,  # probability apply mixup each batch

    # ---- MC DROPOUT ----
    "use_mc_dropout": True,
    "mc_dropout_samples": 20,
    "mc_dropout_rate": 0.15,  # applied at feature level dropout layer (only active in MC sampling)

    # ---- temperature scaling ----
    "use_temperature_scaling": True,
    "temp_lr": 0.05,
    "temp_steps": 300,
    "use_binned_temperature": True,
    "n_temp_bins": 10,

    # ---- threshold calibration ----
    "threshold_calibration": True,
    "threshold_min": 0.05,
    "threshold_max": 0.95,
    "threshold_step": 0.01,

    # ---- domain-specific ----
    "use_domain_adjustment": True,
    "sensitive_domains": ["ngân hàng", "vietbank", "cháy", "who", "y tế", "công an", "bệnh viện"],
    "min_sensitive_val_samples": 30,  # if too few sensitive samples -> fallback to global thr
}

# ============================================================
# OUTPUT DIR + SAVE UTILS
# ============================================================

OUT_DIR = Path(CONFIG["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def json_safe(x):
    if isinstance(x, Path):
        return str(x)
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)

save_json(OUT_DIR / "config.json", {k: json_safe(v) for k, v in CONFIG.items()})

print("=" * 100)
print("VIETNAMESE FAKE NEWS - CROSS-ATTENTION (FIXED)")
print("=" * 100)
print(f"Device: {CONFIG['device']}")
print("Label mapping: 0=REAL, 1=FAKE")
print(f"Dataset: {CONFIG['data_path']}")
print(f"Output dir: {OUT_DIR}")
print(f"MC Dropout: {CONFIG['use_mc_dropout']} samples={CONFIG['mc_dropout_samples']} rate={CONFIG['mc_dropout_rate']}")
print(f"Focal: {CONFIG['use_focal_loss']} | Mixup: {CONFIG['use_mixup']} | ConfPenalty: {CONFIG['use_confidence_penalty']}")
print(f"TempScaling: {CONFIG['use_temperature_scaling']} | Binned: {CONFIG['use_binned_temperature']}")
print(f"Domain adjustment: {CONFIG['use_domain_adjustment']}")
print("=" * 100)

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

set_seed(CONFIG["seed"])

# ============================================================
# CLEAN TEXT
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
    if pd.isna(text):
        return ""
    text = str(text)
    text = emoji_to_text(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid(text):
    return len(text.split()) >= 8

def is_sensitive_domain(text: str) -> bool:
    if not CONFIG.get("use_domain_adjustment", False):
        return False
    tl = text.lower()
    return any(k in tl for k in CONFIG.get("sensitive_domains", []))

# ============================================================
# Union-Find for clustering near-duplicates
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def build_near_dup_groups(texts, threshold=0.92, k=20, ngram_range=(4,6), min_df=2):
    n = len(texts)
    if n == 0:
        return np.array([], dtype=int)

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        dtype=np.float32
    )
    X = vec.fit_transform(texts)

    nnm = NearestNeighbors(
        n_neighbors=min(k, n),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    nnm.fit(X)
    dists, idxs = nnm.kneighbors(X, return_distance=True)

    uf = UnionFind(n)
    for i in range(n):
        for dist, j in zip(dists[i], idxs[i]):
            if j == i:
                continue
            sim = 1.0 - float(dist)
            if sim >= threshold:
                uf.union(i, int(j))

    roots = np.array([uf.find(i) for i in range(n)], dtype=int)
    uniq = {}
    gid = np.zeros(n, dtype=int)
    c = 0
    for i, r in enumerate(roots):
        if r not in uniq:
            uniq[r] = c
            c += 1
        gid[i] = uniq[r]
    return gid

def report_leak_exact(a_texts, b_texts, name):
    sa = set(a_texts)
    sb = set(b_texts)
    inter = len(sa & sb)
    print(f"Leak exact {name}: {inter}")
    return inter

def report_leak_near(a_texts, b_texts, threshold=0.92, ngram_range=(4,6), min_df=2):
    if len(a_texts)==0 or len(b_texts)==0:
        return {"mean": 0, "median": 0, "p95": 0, "max": 0, "count_ge_thr": 0}

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df, dtype=np.float32)
    X_a = vec.fit_transform(a_texts)
    X_b = vec.transform(b_texts)

    nnm = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(X_a)
    dists, _ = nnm.kneighbors(X_b, return_distance=True)
    sims = 1.0 - dists.reshape(-1)

    stats = {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p95": float(np.quantile(sims, 0.95)),
        "max": float(np.max(sims)),
        "count_ge_thr": int(np.sum(sims >= threshold))
    }
    return stats

# ============================================================
# LOAD DATA + CLEAN + DEDUP
# ============================================================

print("\n" + "="*100)
print("LOADING DATA")
print("="*100)

if not os.path.exists(CONFIG["data_path"]):
    raise FileNotFoundError(f"Not found: {CONFIG['data_path']}")

df = pd.read_csv(CONFIG["data_path"])

if "text" not in df.columns:
    for c in ["content", "article", "news", "body", "title"]:
        if c in df.columns:
            df["text"] = df[c]
            break
if "text" not in df.columns:
    raise ValueError("Cannot find text column")

if "label" not in df.columns:
    for c in ["class", "category", "y"]:
        if c in df.columns:
            df["label"] = df[c]
            break
if "label" not in df.columns:
    raise ValueError("Cannot find label column")

df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

bad = df[~df["label"].isin([0, 1])]
if len(bad) > 0:
    raise ValueError(f"Found labels not in {{0,1}}. Examples:\n{bad.head()}")

print("Cleaning + exact dedup...")
df["text_clean"] = df["text"].apply(clean_text)
df = df[df["text_clean"].apply(is_valid)].copy()
before = len(df)
df = df.drop_duplicates(subset=["text_clean"], keep="first").reset_index(drop=True)
print(f"After exact dedup: {len(df)} (removed {before-len(df)})")

n = len(df)
c0 = int((df["label"]==0).sum())
c1 = int((df["label"]==1).sum())
print(f"Final: {n} samples | REAL={c0} ({c0/n:.1%}) | FAKE={c1} ({c1/n:.1%})")
df.to_csv(OUT_DIR / "dataset_clean_dedup.csv", index=False, encoding="utf-8-sig")

# ============================================================
# BUILD NEAR-DUP GROUPS
# ============================================================

print("\n" + "="*100)
print("BUILDING NEAR-DUP CLUSTERS")
print("="*100)

groups = build_near_dup_groups(
    df["text_clean"].tolist(),
    threshold=float(CONFIG["near_dup_threshold"]),
    k=int(CONFIG["near_dup_k"]),
    ngram_range=tuple(CONFIG["char_ngram_range"]),
    min_df=int(CONFIG["min_df"])
)
df["group"] = groups

n_groups = int(df["group"].nunique())
group_sizes = df["group"].value_counts()
print(f"Groups: {n_groups} | Largest group size: {int(group_sizes.max())}")
print(f"Top 5 group sizes:\n{group_sizes.head(5).to_string()}")
df.to_csv(OUT_DIR / "dataset_with_groups.csv", index=False, encoding="utf-8-sig")

# ============================================================
# STRATIFIED GROUP SPLIT
# ============================================================

print("\n" + "="*100)
print("LEAK-SAFE SPLIT (STRATIFIED BY LABEL, GROUP-AWARE)")
print("="*100)

y = df["label"].values
g = df["group"].values

kfold = 10
sgkf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=CONFIG["seed"])

best_fold = None
best_diff = 1e9
test_target = float(CONFIG["test_ratio"])

splits = list(sgkf.split(df, y, groups=g))
for fold_i, (trainval_idx, test_idx) in enumerate(splits):
    ratio = len(test_idx)/len(df)
    diff = abs(ratio - test_target)
    if diff < best_diff:
        best_diff = diff
        best_fold = (trainval_idx, test_idx, ratio, fold_i)

trainval_idx, test_idx, ratio, fold_i = best_fold
print(f"Picked fold {fold_i} for TEST: size={len(test_idx)} ({ratio:.3f})")

df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

val_target = float(CONFIG["val_ratio_of_trainval"])
kfold2 = 9
sgkf2 = StratifiedGroupKFold(n_splits=kfold2, shuffle=True, random_state=CONFIG["seed"]+7)
splits2 = list(sgkf2.split(df_trainval, df_trainval["label"].values, groups=df_trainval["group"].values))

best2 = None
best2_diff = 1e9
for fold_i2, (train_idx, val_idx) in enumerate(splits2):
    ratio2 = len(val_idx)/len(df_trainval)
    diff2 = abs(ratio2 - val_target)
    if diff2 < best2_diff:
        best2_diff = diff2
        best2 = (train_idx, val_idx, ratio2, fold_i2)

train_idx, val_idx, ratio2, fold_i2 = best2
print(f"Picked fold {fold_i2} for VAL: size={len(val_idx)} ({ratio2:.3f})")

train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
val_df   = df_trainval.iloc[val_idx].reset_index(drop=True)
test_df  = df_test.copy()

def dist_print(name, dfx):
    n_ = len(dfx)
    c0_ = int((dfx["label"]==0).sum())
    c1_ = int((dfx["label"]==1).sum())
    print(f"{name}: {n_} | REAL={c0_} ({c0_/n_:.1%}) | FAKE={c1_} ({c1_/n_:.1%}) | groups={dfx['group'].nunique()}")

dist_print("Train", train_df)
dist_print("Val  ", val_df)
dist_print("Test ", test_df)

overlap_tv = set(train_df["group"]) & set(val_df["group"])
overlap_tt = set(train_df["group"]) & set(test_df["group"])
overlap_vt = set(val_df["group"]) & set(test_df["group"])
print(f"\nGroup overlap Train∩Val={len(overlap_tv)} | Train∩Test={len(overlap_tt)} | Val∩Test={len(overlap_vt)}")

train_df.to_csv(OUT_DIR / "train_split.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(OUT_DIR / "val_split.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(OUT_DIR / "test_split.csv", index=False, encoding="utf-8-sig")

# ============================================================
# LEAK REPORT
# ============================================================

print("\n" + "="*100)
print("LEAK REPORT")
print("="*100)

report_leak_exact(train_df["text_clean"], val_df["text_clean"], "Train∩Val")
report_leak_exact(train_df["text_clean"], test_df["text_clean"], "Train∩Test")
report_leak_exact(val_df["text_clean"], test_df["text_clean"], "Val∩Test")

stats_tv = report_leak_near(train_df["text_clean"].tolist(), val_df["text_clean"].tolist(),
                            threshold=CONFIG["near_dup_threshold"],
                            ngram_range=CONFIG["char_ngram_range"],
                            min_df=CONFIG["min_df"])
stats_tt = report_leak_near(train_df["text_clean"].tolist(), test_df["text_clean"].tolist(),
                            threshold=CONFIG["near_dup_threshold"],
                            ngram_range=CONFIG["char_ngram_range"],
                            min_df=CONFIG["min_df"])
stats_vt = report_leak_near(val_df["text_clean"].tolist(), test_df["text_clean"].tolist(),
                            threshold=CONFIG["near_dup_threshold"],
                            ngram_range=CONFIG["char_ngram_range"],
                            min_df=CONFIG["min_df"])

leak_stats = {"val_to_train": stats_tv, "test_to_train": stats_tt, "test_to_val": stats_vt}
save_json(OUT_DIR / "leak_stats.json", leak_stats)

print("\nNear-dup cosine stats (char-ngram TFIDF):")
print("Val->Train:", stats_tv)
print("Test->Train:", stats_tt)
print("Test->Val:", stats_vt)

# ============================================================
# TOKENIZERS + DATASET
# ============================================================

tokE = AutoTokenizer.from_pretrained(CONFIG["electra_name"], use_fast=False)
tokP = AutoTokenizer.from_pretrained(CONFIG["phobert_name"], use_fast=False)

class DualTokDataset(Dataset):
    def __init__(self, texts, labels, tokE, tokP, maxE=256, maxP=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokE = tokE
        self.tokP = tokP
        self.maxE = int(maxE)
        self.maxP = int(maxP)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = int(self.labels[idx])

        e = self.tokE(text, max_length=self.maxE, padding="max_length", truncation=True, return_tensors="pt")
        p = self.tokP(text, max_length=self.maxP, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "e_input_ids": e["input_ids"].squeeze(0).long(),
            "e_attn": e["attention_mask"].squeeze(0).long(),
            "p_input_ids": p["input_ids"].squeeze(0).long(),
            "p_attn": p["attention_mask"].squeeze(0).long(),
            "label": torch.tensor(y, dtype=torch.long),
        }

def make_dual_loaders(bs):
    train_loader = DataLoader(
        DualTokDataset(train_df["text_clean"].values, train_df["label"].values, tokE, tokP,
                       CONFIG["max_length_electra"], CONFIG["max_length_phobert"]),
        batch_size=int(bs), shuffle=True
    )
    val_loader = DataLoader(
        DualTokDataset(val_df["text_clean"].values, val_df["label"].values, tokE, tokP,
                       CONFIG["max_length_electra"], CONFIG["max_length_phobert"]),
        batch_size=int(bs), shuffle=False
    )
    test_loader = DataLoader(
        DualTokDataset(test_df["text_clean"].values, test_df["label"].values, tokE, tokP,
                       CONFIG["max_length_electra"], CONFIG["max_length_phobert"]),
        batch_size=int(bs), shuffle=False
    )
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = make_dual_loaders(CONFIG["batch_size"])

def compute_class_weights(y_arr):
    counts = np.bincount(y_arr, minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    return counts, torch.tensor(weights, dtype=torch.float32, device=CONFIG["device"])

# ============================================================
# MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.size(0)
    index = torch.randperm(bs).to(x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam

# ============================================================
# MODEL
# ============================================================

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def unfreeze_last_n_layers_electra(electra_model, n_last: int):
    if n_last <= 0: return
    if not hasattr(electra_model, "encoder"): return
    layers = electra_model.encoder.layer
    L = len(layers)
    for i in range(max(0, L - n_last), L):
        for p in layers[i].parameters():
            p.requires_grad = True

def unfreeze_last_n_layers_roberta(roberta_model, n_last: int):
    if n_last <= 0: return
    if not hasattr(roberta_model, "encoder"): return
    layers = roberta_model.encoder.layer
    L = len(layers)
    for i in range(max(0, L - n_last), L):
        for p in layers[i].parameters():
            p.requires_grad = True

class CrossAttnFusionClassifier(nn.Module):
    def __init__(self, electra_name, phobert_name, dropout=0.3, heads=8,
                 cross_attn_dropout=0.1, mc_dropout_rate=0.15):
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

        self.norm = nn.LayerNorm(Hp)
        self.dropout = nn.Dropout(dropout)

        # MC dropout layer (disabled in eval normally; enabled only for MC sampling)
        self.mc_dropout = nn.Dropout(p=mc_dropout_rate)

        self.head = nn.Sequential(
            nn.Linear(Hp * 2, Hp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(Hp, 2)
        )

    def fuse_features(self, e_input_ids, e_attn, p_input_ids, p_attn):
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
        return feat

    def forward(self, e_input_ids, e_attn, p_input_ids, p_attn, return_features=False, use_mc_dropout=False):
        feat = self.fuse_features(e_input_ids, e_attn, p_input_ids, p_attn)
        if return_features:
            return feat

        # normal dropout for training/inference
        if use_mc_dropout:
            feat = self.mc_dropout(feat)   # only used in MC sampling
        else:
            feat = self.dropout(feat)

        logits = self.head(feat)
        return logits

model = CrossAttnFusionClassifier(
    CONFIG["electra_name"],
    CONFIG["phobert_name"],
    dropout=float(CONFIG["dropout"]),
    heads=int(CONFIG["cross_attn_heads"]),
    cross_attn_dropout=float(CONFIG["cross_attn_dropout"]),
    mc_dropout_rate=float(CONFIG.get("mc_dropout_rate", 0.15)),
).to(CONFIG["device"])

if CONFIG["freeze_backbones"]:
    set_requires_grad(model.e, False)
    set_requires_grad(model.p, False)
    if int(CONFIG["unfreeze_last_n_layers_electra"]) > 0:
        unfreeze_last_n_layers_electra(model.e, int(CONFIG["unfreeze_last_n_layers_electra"]))
        print(f"✅ Unfroze last {CONFIG['unfreeze_last_n_layers_electra']} layers of ELECTRA")
    if int(CONFIG["unfreeze_last_n_layers_phobert"]) > 0:
        unfreeze_last_n_layers_roberta(model.p, int(CONFIG["unfreeze_last_n_layers_phobert"]))
        print(f"✅ Unfroze last {CONFIG['unfreeze_last_n_layers_phobert']} layers of PhoBERT")

    set_requires_grad(model.proj_kv, True)
    set_requires_grad(model.cross_attn, True)
    set_requires_grad(model.norm, True)
    set_requires_grad(model.dropout, True)
    set_requires_grad(model.mc_dropout, True)
    set_requires_grad(model.head, True)

# ============================================================
# LOSSES
# ============================================================

y_train = train_df["label"].values.astype(int)
counts, class_w = compute_class_weights(y_train)
print("\nClass counts [REAL, FAKE]:", counts, "| class_weights:", class_w.detach().cpu().numpy())

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

class ConfidencePenalty(nn.Module):
    def __init__(self, beta=0.05):
        super().__init__()
        self.beta = beta

    def forward(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return self.beta * (-entropy.mean())

def label_smoothing_ce(logits, targets, class_w, smoothing=0.0):
    if smoothing <= 0.0:
        return F.cross_entropy(logits, targets, weight=class_w)

    num_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    loss = -(true_dist * log_probs) * class_w.unsqueeze(0)
    return loss.sum(dim=1).mean()

focal_loss_fn = FocalLoss(alpha=float(CONFIG["focal_alpha"]),
                          gamma=float(CONFIG["focal_gamma"]),
                          class_weights=class_w) if CONFIG.get("use_focal_loss", False) else None
conf_penalty_fn = ConfidencePenalty(beta=float(CONFIG["confidence_penalty_beta"])) if CONFIG.get("use_confidence_penalty", False) else None
LS = float(CONFIG.get("label_smoothing", 0.0))

def criterion_fn(logits, labels, mixup_labels=None, mixup_lambda=None):
    if focal_loss_fn is not None:
        if mixup_labels is not None:
            y_a, y_b = mixup_labels
            loss = mixup_lambda * focal_loss_fn(logits, y_a) + (1 - mixup_lambda) * focal_loss_fn(logits, y_b)
        else:
            loss = focal_loss_fn(logits, labels)
    else:
        if mixup_labels is not None:
            y_a, y_b = mixup_labels
            loss = mixup_lambda * label_smoothing_ce(logits, y_a, class_w, LS) + (1 - mixup_lambda) * label_smoothing_ce(logits, y_b, class_w, LS)
        else:
            loss = label_smoothing_ce(logits, labels, class_w, LS)

    if conf_penalty_fn is not None:
        loss = loss + conf_penalty_fn(logits)
    return loss

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=float(CONFIG["learning_rate"]),
    weight_decay=float(CONFIG["weight_decay"]),
)

total_steps = max(len(train_loader) * int(CONFIG["epochs"]), 1)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(CONFIG["warmup_steps"]),
    num_training_steps=total_steps
)

# ============================================================
# CALIBRATION METRICS
# ============================================================

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)

# ============================================================
# COLLECT LOGITS
# ============================================================

@torch.no_grad()
def collect_logits_and_labels(loader):
    model.eval()
    all_logits = []
    all_y = []
    for batch in tqdm(loader, desc="Collect logits", leave=False):
        e_input_ids = batch["e_input_ids"].to(CONFIG["device"])
        e_attn      = batch["e_attn"].to(CONFIG["device"])
        p_input_ids = batch["p_input_ids"].to(CONFIG["device"])
        p_attn      = batch["p_attn"].to(CONFIG["device"])
        y           = batch["label"].to(CONFIG["device"])

        logits = model(e_input_ids, e_attn, p_input_ids, p_attn, use_mc_dropout=False)
        all_logits.append(logits.detach())
        all_y.append(y.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)

# ============================================================
# TEMPERATURE SCALING (SINGLE + BINNED)
# ============================================================

def fit_temperature_on_val(val_loader):
    logits, y = collect_logits_and_labels(val_loader)
    logits = logits.to(CONFIG["device"])
    y = y.to(CONFIG["device"])

    log_T = torch.nn.Parameter(torch.zeros((), device=CONFIG["device"]))
    opt = torch.optim.Adam([log_T], lr=float(CONFIG["temp_lr"]))

    for _ in range(int(CONFIG["temp_steps"])):
        opt.zero_grad(set_to_none=True)
        T = torch.exp(log_T).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        opt.step()

    return float(torch.exp(log_T).clamp(min=1e-3, max=100.0).detach().cpu().item())

def fit_binned_temperature_on_val(val_loader, n_bins=10):
    logits, y = collect_logits_and_labels(val_loader)
    logits = logits.to(CONFIG["device"])
    y = y.to(CONFIG["device"])

    # bins based on confidence from unscaled logits (fixed)
    with torch.no_grad():
        probs0 = F.softmax(logits, dim=1)
        conf0 = torch.max(probs0, dim=1)[0]
        bin_idx = (conf0 * n_bins).long().clamp(0, n_bins-1)

    log_temps = nn.Parameter(torch.zeros(n_bins, device=CONFIG["device"]))
    opt = torch.optim.Adam([log_temps], lr=float(CONFIG["temp_lr"]))

    print(f"\n🔥 Fitting binned temperature ({n_bins} bins)...")
    for step in range(int(CONFIG["temp_steps"])):
        opt.zero_grad(set_to_none=True)
        temps = torch.exp(log_temps).clamp(min=0.5, max=10.0)  # safer range than 0.1
        T_sample = temps[bin_idx].unsqueeze(1)
        loss = F.cross_entropy(logits / T_sample, y)
        loss.backward()
        opt.step()
        if (step+1) % 100 == 0:
            print(f"  Step {step+1}/{CONFIG['temp_steps']}: loss={loss.item():.4f}")

    temps_final = torch.exp(log_temps).clamp(min=0.5, max=10.0).detach().cpu().numpy()
    print("✅ Binned temperatures:", temps_final)
    return temps_final

@torch.no_grad()
def probs_from_logits_calibrated(logits, calib):
    """
    ✅ SINGLE SOURCE OF TRUTH calibration
    calib = {"T": float|None, "bins": list|None}
    """
    if calib.get("bins") is not None:
        temps_bins = calib["bins"]
        probs0 = F.softmax(logits, dim=1)
        conf0 = torch.max(probs0, dim=1)[0]
        n_bins = len(temps_bins)
        idx = (conf0 * n_bins).long().clamp(0, n_bins-1)
        temps = torch.tensor(temps_bins, device=logits.device, dtype=logits.dtype).clamp(min=0.5)
        T_sample = temps[idx].unsqueeze(1)
        return F.softmax(logits / T_sample, dim=1)
    elif calib.get("T") is not None:
        T = torch.tensor(float(calib["T"]), device=logits.device, dtype=logits.dtype).clamp(min=1e-3)
        return F.softmax(logits / T, dim=1)
    else:
        return F.softmax(logits, dim=1)

def find_best_threshold_from_probs(p_fake, y_true, thr_min=0.05, thr_max=0.95, thr_step=0.01):
    best_thr, best_f1 = 0.5, -1.0
    thr = thr_min
    while thr <= thr_max + 1e-12:
        pred = (p_fake >= thr).astype(int)
        f1m = f1_score(y_true, pred, average="macro")
        if f1m > best_f1:
            best_f1 = float(f1m)
            best_thr = float(thr)
        thr += thr_step
    return best_thr, best_f1

# ============================================================
# EVAL
# ============================================================

@torch.no_grad()
def eval_with_calib(loader, y_true, calib, threshold):
    model.eval()
    probs_all = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        e_input_ids = batch["e_input_ids"].to(CONFIG["device"])
        e_attn      = batch["e_attn"].to(CONFIG["device"])
        p_input_ids = batch["p_input_ids"].to(CONFIG["device"])
        p_attn      = batch["p_attn"].to(CONFIG["device"])
        logits = model(e_input_ids, e_attn, p_input_ids, p_attn, use_mc_dropout=False)
        probs = probs_from_logits_calibrated(logits, calib)
        probs_all.append(probs.cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    p_fake = probs_all[:, 1]
    pred = (p_fake >= float(threshold)).astype(int)

    acc = accuracy_score(y_true, pred)
    f1m = f1_score(y_true, pred, average="macro")
    ece = compute_ece(probs_all, y_true)

    return {
        "acc": float(acc),
        "f1_macro": float(f1m),
        "ece": float(ece),
        "probs": probs_all,
        "pred": pred
    }

# ============================================================
# MC DROPOUT PREDICT (fixed)
# ============================================================

def enable_dropout_layers(m):
    # set only dropout modules to train mode
    for module in m.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.no_grad()
def mc_probs_one_batch(e_input_ids, e_attn, p_input_ids, p_attn, n_samples=20, calib=None):
    """
    Run MC dropout sampling by enabling dropout layers temporarily.
    Returns mean_probs [B,2] and uncertainty [B]
    """
    model.eval()
    preds = []
    for _ in range(n_samples):
        enable_dropout_layers(model)  # only dropout active
        logits = model(e_input_ids, e_attn, p_input_ids, p_attn, use_mc_dropout=True)
        probs = probs_from_logits_calibrated(logits, calib or {"T": None, "bins": None})
        preds.append(probs)

    preds = torch.stack(preds, dim=0)  # [S,B,2]
    mean_probs = preds.mean(dim=0)
    var = preds.var(dim=0).sum(dim=1)  # [B]
    return mean_probs, var

# ============================================================
# CHECKPOINT + CALIB LOAD/SAVE
# ============================================================

def load_checkpoint_and_calibration(run_dir: str, device):
    run_dir = Path(run_dir)
    ckpt = run_dir / "best_model_state_dict.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("✅ Loaded model:", ckpt)

    calib = {"T": None, "bins": None}
    thr_pack = {"thr_global": 0.5, "thr_normal": 0.5, "thr_sensitive": 0.5}

    t_path = run_dir / "calibration_temperature.json"
    bins_path = run_dir / "calibration_binned_temps.json"
    thr_path = run_dir / "calibration_thresholds.json"

    if bins_path.exists():
        data = json.load(open(bins_path, "r", encoding="utf-8"))
        calib["bins"] = data["binned_temperatures"]
        print("✅ Loaded binned temperatures")
    elif t_path.exists():
        calib["T"] = float(json.load(open(t_path, "r", encoding="utf-8"))["temperature"])
        print("✅ Loaded single temperature")

    if thr_path.exists():
        thr_pack = json.load(open(thr_path, "r", encoding="utf-8"))
        print("✅ Loaded thresholds:", thr_pack)

    return calib, thr_pack

# ============================================================
# TRAIN
# ============================================================

CALIB = {"T": None, "bins": None}
THR = {"thr_global": 0.5, "thr_normal": 0.5, "thr_sensitive": 0.5}

if CONFIG["mode"] == "train":
    print("\n" + "="*100)
    print("TRAINING")
    print("="*100)

    best_state = None
    best_f1 = -1.0
    history = []

    for ep in range(int(CONFIG["epochs"])):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train ep{ep+1}", leave=False):
            e_input_ids = batch["e_input_ids"].to(CONFIG["device"])
            e_attn      = batch["e_attn"].to(CONFIG["device"])
            p_input_ids = batch["p_input_ids"].to(CONFIG["device"])
            p_attn      = batch["p_attn"].to(CONFIG["device"])
            labels      = batch["label"].to(CONFIG["device"])

            optimizer.zero_grad(set_to_none=True)

            # Feature extraction
            feat = model(e_input_ids, e_attn, p_input_ids, p_attn, return_features=True)

            # Mixup on features (stable)
            if CONFIG.get("use_mixup", False) and random.random() < float(CONFIG.get("mixup_prob", 0.5)):
                feat_m, y_a, y_b, lam = mixup_data(feat, labels, alpha=float(CONFIG["mixup_alpha"]))
                feat_m = model.dropout(feat_m)  # normal dropout in train
                logits = model.head(feat_m)
                loss = criterion_fn(logits, labels, mixup_labels=(y_a, y_b), mixup_lambda=lam)
            else:
                feat_d = model.dropout(feat)
                logits = model.head(feat_d)
                loss = criterion_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["grad_clip"]))
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())

        # validate (no calibration here, raw probs @thr=0.5 just for early stop)
        y_val = val_df["label"].values.astype(int)
        val_raw = eval_with_calib(val_loader, y_val, calib={"T": None, "bins": None}, threshold=0.5)

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {ep+1}/{CONFIG['epochs']} | loss={avg_loss:.4f} | "
              f"val_acc={val_raw['acc']*100:.2f}% | val_macroF1={val_raw['f1_macro']:.4f} | val_ECE={val_raw['ece']:.4f}")

        history.append({
            "epoch": int(ep+1),
            "train_loss": float(avg_loss),
            "val_acc_raw": float(val_raw["acc"]),
            "val_macro_f1_raw": float(val_raw["f1_macro"]),
            "val_ece_raw": float(val_raw["ece"]),
        })
        save_json(OUT_DIR / "train_history.json", history)

        if val_raw["f1_macro"] > best_f1:
            best_f1 = val_raw["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            print("✨ New best model!")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), OUT_DIR / "best_model_state_dict.pt")
    print(f"\n✅ Saved best model -> {OUT_DIR / 'best_model_state_dict.pt'}")

    # ============================================================
    # CALIBRATION
    # ============================================================

    if bool(CONFIG.get("use_temperature_scaling", True)):
        print("\n" + "="*100)
        print("TEMPERATURE SCALING")
        print("="*100)

        if CONFIG.get("use_binned_temperature", False):
            bins = fit_binned_temperature_on_val(val_loader, n_bins=int(CONFIG.get("n_temp_bins", 10)))
            CALIB = {"T": None, "bins": bins.tolist()}
            save_json(OUT_DIR / "calibration_binned_temps.json", {
                "binned_temperatures": CALIB["bins"],
                "n_bins": len(CALIB["bins"])
            })
        else:
            T = fit_temperature_on_val(val_loader)
            CALIB = {"T": float(T), "bins": None}
            print(f"✅ Learned temperature T = {T}")
            save_json(OUT_DIR / "calibration_temperature.json", {"temperature": float(T)})
    else:
        CALIB = {"T": None, "bins": None}

    # collect calibrated probs on VAL for thresholds
    print("\n" + "="*100)
    print("THRESHOLD CALIBRATION (GLOBAL + DOMAIN)")
    print("="*100)

    logits_val, yv_t = collect_logits_and_labels(val_loader)
    probs_val = probs_from_logits_calibrated(logits_val.to(CONFIG["device"]), CALIB).cpu().numpy()
    p_fake_val = probs_val[:, 1]
    y_val_np = yv_t.cpu().numpy().astype(int)

    thr_global, f1_global = find_best_threshold_from_probs(
        p_fake_val, y_val_np,
        thr_min=float(CONFIG["threshold_min"]),
        thr_max=float(CONFIG["threshold_max"]),
        thr_step=float(CONFIG["threshold_step"])
    )
    THR["thr_global"] = float(thr_global)

    # Domain thresholds
    if CONFIG.get("use_domain_adjustment", False):
        val_texts = val_df["text_clean"].values
        mask_sensitive = np.array([is_sensitive_domain(t) for t in val_texts], dtype=bool)
        mask_normal = ~mask_sensitive

        # normal
        if mask_normal.sum() > 10:
            thr_n, f1_n = find_best_threshold_from_probs(
                p_fake_val[mask_normal], y_val_np[mask_normal],
                thr_min=float(CONFIG["threshold_min"]),
                thr_max=float(CONFIG["threshold_max"]),
                thr_step=float(CONFIG["threshold_step"])
            )
        else:
            thr_n, f1_n = thr_global, f1_global

        # sensitive
        if mask_sensitive.sum() >= int(CONFIG.get("min_sensitive_val_samples", 30)):
            thr_s, f1_s = find_best_threshold_from_probs(
                p_fake_val[mask_sensitive], y_val_np[mask_sensitive],
                thr_min=float(CONFIG["threshold_min"]),
                thr_max=float(CONFIG["threshold_max"]),
                thr_step=float(CONFIG["threshold_step"])
            )
        else:
            thr_s, f1_s = thr_global, f1_global

        THR["thr_normal"] = float(thr_n)
        THR["thr_sensitive"] = float(thr_s)

        print(f"VAL sensitive samples: {int(mask_sensitive.sum())} | normal: {int(mask_normal.sum())}")
        print(f"Best thr_global   = {thr_global:.3f} | val_macroF1={f1_global:.4f}")
        print(f"Best thr_normal   = {thr_n:.3f} | val_macroF1={f1_n:.4f}")
        print(f"Best thr_sensitive= {thr_s:.3f} | val_macroF1={f1_s:.4f}")
    else:
        THR["thr_normal"] = float(thr_global)
        THR["thr_sensitive"] = float(thr_global)
        print(f"Best thr_global = {thr_global:.3f} | val_macroF1={f1_global:.4f}")

    save_json(OUT_DIR / "calibration_thresholds.json", THR)
    print("✅ Saved thresholds ->", OUT_DIR / "calibration_thresholds.json")
    print("✅ Calibration summary:", {"CALIB": CALIB, "THR": THR})

elif CONFIG["mode"] == "infer":
    if not CONFIG["load_run_dir"]:
        raise ValueError("Infer mode requires CONFIG['load_run_dir'] to be set.")
    CALIB, THR = load_checkpoint_and_calibration(CONFIG["load_run_dir"], CONFIG["device"])
else:
    raise ValueError("CONFIG['mode'] must be 'train' or 'infer'")

# ============================================================
# FINAL EVAL ON TEST (USING SAME CALIB + THR_GLOBAL)
# ============================================================

print("\n" + "="*100)
print("FINAL EVALUATION ON TEST (CALIB + thr_global)")
print("="*100)

y_test = test_df["label"].values.astype(int)
test_eval = eval_with_calib(test_loader, y_test, CALIB, threshold=THR["thr_global"])

pred_test = test_eval["pred"]
acc = test_eval["acc"]
f1_fake = f1_score(y_test, pred_test, pos_label=1, average="binary")
f1_real = f1_score(y_test, pred_test, pos_label=0, average="binary")
macro = (f1_fake + f1_real) / 2

print(f"Accuracy:    {acc*100:.2f}%")
print(f"Macro-F1:    {macro:.4f}")
print(f"F1 FAKE(1):  {f1_fake:.4f}")
print(f"F1 REAL(0):  {f1_real:.4f}")
print(f"Gap:         {abs(f1_fake - f1_real):.4f}")
print(f"ECE:         {test_eval['ece']:.4f}")
print("\nClassification Report:")
report_str = classification_report(y_test, pred_test, target_names=["REAL (0)", "FAKE (1)"], digits=4)
print(report_str)

save_json(OUT_DIR / "test_metrics.json", {
    "accuracy": acc,
    "macro_f1": float(macro),
    "f1_fake": float(f1_fake),
    "f1_real": float(f1_real),
    "gap": float(abs(f1_fake - f1_real)),
    "ece": float(test_eval["ece"]),
    "thresholds": THR,
    "calib": CALIB,
})
save_text(OUT_DIR / "classification_report.txt", report_str)
print(f"\n✅ Saved test metrics -> {OUT_DIR}")

# ============================================================
# PREDICT_ONE (FIXED: same calibration + domain thresholds)
# ============================================================

@torch.no_grad()
def predict_one(text: str, use_mc=None):
    text = clean_text(text)
    if not is_valid(text):
        return {"ok": False, "error": "Text too short after cleaning", "text": text}

    if use_mc is None:
        use_mc = bool(CONFIG.get("use_mc_dropout", False))

    encE = tokE(text, max_length=CONFIG["max_length_electra"],
                padding="max_length", truncation=True, return_tensors="pt")
    encP = tokP(text, max_length=CONFIG["max_length_phobert"],
                padding="max_length", truncation=True, return_tensors="pt")

    e_input_ids = encE["input_ids"].to(CONFIG["device"])
    e_attn      = encE["attention_mask"].to(CONFIG["device"])
    p_input_ids = encP["input_ids"].to(CONFIG["device"])
    p_attn      = encP["attention_mask"].to(CONFIG["device"])

    sensitive = is_sensitive_domain(text)
    thr = THR["thr_sensitive"] if sensitive else THR["thr_normal"]

    if use_mc:
        mean_probs, unc = mc_probs_one_batch(
            e_input_ids, e_attn, p_input_ids, p_attn,
            n_samples=int(CONFIG.get("mc_dropout_samples", 20)),
            calib=CALIB
        )
        probs = mean_probs.squeeze(0)
        uncertainty = float(unc.squeeze(0).cpu().item())
    else:
        model.eval()
        logits = model(e_input_ids, e_attn, p_input_ids, p_attn, use_mc_dropout=False)
        probs = probs_from_logits_calibrated(logits, CALIB).squeeze(0)
        uncertainty = None

    p_fake = float(probs[1].cpu().item())
    pred = 1 if p_fake >= float(thr) else 0
    conf = float(torch.max(probs).cpu().item())

    out = {
        "ok": True,
        "pred": int(pred),
        "p_fake": float(p_fake),
        "conf": float(conf),
        "thr_used": float(thr),
        "is_sensitive_domain": bool(sensitive),
        "text": text,
        "calib_used": {"has_bins": CALIB.get("bins") is not None, "T": CALIB.get("T", None)},
    }
    if uncertainty is not None:
        out["uncertainty"] = float(uncertainty)
    return out

# ============================================================
# PIPELINE TEST (predict_one over test)
# ============================================================

@torch.no_grad()
def evaluate_on_test_predict_one(top_k_errors=12):
    y_true = test_df["label"].values.astype(int)
    texts  = test_df["text_clean"].values

    preds, p_fakes, confs = [], [], []
    for t in tqdm(texts, desc="Test inference (predict_one)"):
        o = predict_one(t, use_mc=False)  # keep deterministic for metric
        if not o["ok"]:
            preds.append(0); p_fakes.append(0.0); confs.append(0.5)
        else:
            preds.append(o["pred"]); p_fakes.append(o["p_fake"]); confs.append(o["conf"])

    preds = np.array(preds, dtype=int)
    p_fakes = np.array(p_fakes, dtype=float)
    confs = np.array(confs, dtype=float)

    acc_ = accuracy_score(y_true, preds)
    f1_fake_ = f1_score(y_true, preds, pos_label=1)
    f1_real_ = f1_score(y_true, preds, pos_label=0)
    macro_ = (f1_fake_ + f1_real_) / 2

    print("\n" + "="*100)
    print("TEST SET METRICS (predict_one pipeline) ✅ should MATCH final eval trend")
    print("="*100)
    print(f"Accuracy: {acc_*100:.2f}%")
    print(f"Macro-F1: {macro_:.4f} | F1_FAKE={f1_fake_:.4f} | F1_REAL={f1_real_:.4f} | Gap={abs(f1_fake_-f1_real_):.4f}")

    cm = confusion_matrix(y_true, preds, labels=[0,1])
    print("\nConfusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)

    wrong = np.where(preds != y_true)[0]
    if len(wrong) == 0:
        print("\n✨ No errors on test set!")
        return

    wrong_sorted = wrong[np.argsort(-confs[wrong])]
    print("\n" + "="*100)
    print(f"TOP {min(top_k_errors, len(wrong_sorted))} WRONG PREDICTIONS (by confidence)")
    print("="*100)

    lines = []
    for i in wrong_sorted[:top_k_errors]:
        snippet = texts[i][:300].replace("\n", " ")
        print(f"\nIDX={i} | TRUE={y_true[i]} | PRED={preds[i]} | p_fake={p_fakes[i]:.3f} | conf={confs[i]:.3f}")
        print(f"TEXT: {snippet}...")
        lines.append(f"IDX={i}\tTRUE={y_true[i]}\tPRED={preds[i]}\tp_fake={p_fakes[i]:.6f}\tconf={confs[i]:.6f}\tTEXT={texts[i].replace(chr(10),' ')[:1200]}")
    save_text(OUT_DIR / "top_wrong_predictions_predict_one.txt", "\n".join(lines))
    print(f"\n✅ Saved wrong predictions -> {OUT_DIR / 'top_wrong_predictions_predict_one.txt'}")

# ============================================================
# GENERALIZATION CHECK
# ============================================================

GENERALIZATION_10 = [
    "TIN SỐC!!! Chỉ cần uống nước chanh theo cách này 3 ngày là khỏi hoàn toàn tiểu đường, bác sĩ cũng bất ngờ. Xem ngay!!!",
    "Sở Giao thông Vận tải TP.HCM thông báo điều chỉnh tổ chức giao thông một số tuyến đường khu vực trung tâm để phục vụ thi công, thời gian áp dụng từ ngày 10/02.",
    "Các nhà khoa học xác nhận người ngoài hành tinh đã hạ cánh ở Việt Nam và để lại thiết bị lạ, video bằng chứng đang lan truyền mạnh.",
    "Giá vàng trong nước sáng nay biến động nhẹ, nhiều doanh nghiệp điều chỉnh tăng/giảm vài chục nghìn đồng mỗi lượng so với cuối ngày hôm qua.",
    "CẢNH BÁO KHẨN: Ai nhận cuộc gọi số lạ đọc 3 số cuối CCCD sẽ bị trừ tiền tài khoản ngay lập tức. Hãy chia sẻ để cứu mọi người!",
    "Công an cho biết đang xác minh thông tin lan truyền trên mạng liên quan đến vụ việc tại một khu dân cư, đồng thời đề nghị người dân không chia sẻ thông tin chưa kiểm chứng.",
    "Không cần vốn, chỉ với điện thoại bạn có thể kiếm 5 triệu/ngày đảm bảo 100%. Ai cũng làm được, đăng ký ngay kẻo lỡ!",
    "Trung tâm dự báo khí tượng thủy văn nhận định trong vài ngày tới, khu vực Nam Bộ có mưa rào và dông rải rác vào chiều tối.",
    "Bộ Y tế khuyến cáo người dân tiêm nhắc lại vắc-xin theo hướng dẫn và theo dõi thông tin từ các nguồn chính thống khi có dịch bệnh.",
    "Thực hư chuyện một loại nước ngọt đang bị cấm bán vì gây ung thư ngay lập tức? Nhiều người hoang mang, sự thật khiến ai cũng sốc!",
]

def generalization_check_10(samples=GENERALIZATION_10):
    print("\n" + "="*100)
    print("GENERALIZATION CHECK (10 samples) - FIXED")
    print("="*100)
    lines = []
    for i, s in enumerate(samples, 1):
        o = predict_one(s, use_mc=False)
        if not o["ok"]:
            msg = f"{i:02d}. INVALID: {o.get('error')}"
            print(msg); lines.append(msg); continue

        msg = (f"{i:02d}. PRED={o['pred']} | p_fake={o['p_fake']:.3f} | conf={o['conf']:.3f} | "
               f"thr={o['thr_used']:.3f} | sensitive={o['is_sensitive_domain']}\n"
               f"    TEXT: {s}")
        print(msg)
        lines.append(msg)

    save_text(OUT_DIR / "generalization_10_fixed.txt", "\n\n".join(lines))
    print(f"\n✅ Saved generalization results -> {OUT_DIR / 'generalization_10_fixed.txt'}")

print("\n" + "="*100)
print("🚀 READY FOR PREDICTION")
print("="*100)
print("Usage: predict_one('Your text here')")
print("Tip: evaluate_on_test_predict_one() to verify pipeline matches final eval")
print("="*100)

# Auto run extra checks in train mode
if CONFIG["mode"] == "train":
    evaluate_on_test_predict_one(top_k_errors=12)
    generalization_check_10()
