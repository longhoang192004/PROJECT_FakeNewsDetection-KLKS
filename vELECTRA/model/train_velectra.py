# -*- coding: utf-8 -*-
"""
vELECTRA - Vietnamese Fake News Detection
==========================================
Model : FPTAI/velectra-base-discriminator-cased
Data  : data_final/train.csv  (post_message, label)
Split : 80% train / 10% val / 10% test (stratified)
Label : 0 = reliable  |  1 = unreliable
"""

import os, re, random, copy, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm.auto import tqdm

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # --- paths -----------------------------------------------------------
    "data_path": os.path.join(os.path.dirname(__file__), "data_final", "train.csv"),
    "output_dir": os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "velectra_model",
    ),

    # --- model -----------------------------------------------------------
    "model_name": "FPTAI/velectra-base-discriminator-cased",
    "max_length": 256,
    "num_labels": 2,

    # --- training --------------------------------------------------------
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.2,
    "grad_clip": 1.0,

    # --- freezing --------------------------------------------------------
    "freeze_layers": 8,
    "freeze_embeddings": True,

    # --- split -----------------------------------------------------------
    "test_size": 0.10,
    "val_size": 0.10,
    "seed": 42,

    # --- device ----------------------------------------------------------
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ============================================================
# EMOJI → EMOTION TOKEN  MAPPING
# ============================================================

EMOJI_MAP = {
    # --- Vui / Tích cực ---
    "😀": " [VUI] ", "😃": " [VUI] ", "😄": " [VUI] ", "😁": " [VUI] ",
    "😆": " [VUI] ", "😂": " [CƯỜI] ", "🤣": " [CƯỜI] ", "😊": " [VUI] ",
    "😇": " [VUI] ", "🙂": " [VUI] ", "😉": " [TINH_NGHỊCH] ",
    "😍": " [YÊU] ", "🥰": " [YÊU] ", "😘": " [YÊU] ", "😗": " [YÊU] ",
    "😙": " [YÊU] ", "😚": " [YÊU] ", "🥲": " [CẢM_ĐỘNG] ",

    # --- Buồn ---
    "😢": " [BUỒN] ", "😭": " [KHÓC] ", "😞": " [BUỒN] ", "😔": " [BUỒN] ",
    "😟": " [LO_LẮNG] ", "🥺": " [BUỒN] ", "😿": " [BUỒN] ",

    # --- Giận / Tiêu cực ---
    "😠": " [GIẬN] ", "😡": " [GIẬN] ", "🤬": " [GIẬN] ", "😤": " [BỰC] ",
    "👿": " [GIẬN] ", "💢": " [GIẬN] ",

    # --- Ngạc nhiên / Sốc ---
    "😱": " [SỐC] ", "😨": " [SỢ] ", "😰": " [SỢ] ", "😲": " [NGẠC_NHIÊN] ",
    "😮": " [NGẠC_NHIÊN] ", "🤯": " [SỐC] ", "😳": " [NGẠC_NHIÊN] ",

    # --- Sợ ---
    "😧": " [SỢ] ", "😦": " [SỢ] ", "😥": " [SỢ] ",

    # --- Suy nghĩ / Nghi ngờ ---
    "🤔": " [SUY_NGHĨ] ", "🧐": " [SUY_NGHĨ] ", "🤨": " [NGHI_NGỜ] ",

    # --- Cử chỉ ---
    "👍": " [ĐỒNG_Ý] ", "👎": " [PHẢN_ĐỐI] ", "👏": " [KHEN] ",
    "🙏": " [CẦU_NGUYỆN] ", "✌️": " [CHIẾN_THẮNG] ",
    "💪": " [MẠNH_MẼ] ", "🤝": " [BẮT_TAY] ",

    # --- Biểu tượng ---
    "❤️": " [YÊU] ", "💔": " [BUỒN] ", "🔥": " [NÓNG] ", "💯": " [HOÀN_HẢO] ",
    "⚠️": " [CẢNH_BÁO] ", "❌": " [SAI] ", "✅": " [ĐÚNG] ",
    "🚨": " [KHẨN_CẤP] ", "📢": " [THÔNG_BÁO] ", "📌": " [GHI_CHÚ] ",
    "⭐": " [NGÔI_SAO] ", "🌟": " [NGÔI_SAO] ", "💀": " [NGUY_HIỂM] ",
    "🤡": " [HỀ] ", "🐍": " [XẤU_XA] ",

    # --- Cờ / Quốc gia ---
    "🇻🇳": " [VIỆT_NAM] ",

    # --- Misc ---
    "😷": " [BỆNH] ", "🤒": " [BỆNH] ", "🤧": " [BỆNH] ",
    "💉": " [TIÊM_CHỦNG] ", "🦠": " [VIRUS] ", "😴": " [NGỦ] ",
    "🤫": " [IM_LẶNG] ", "🤥": " [NÓI_DỐI] ",
}

# Regex bắt tất cả emoji Unicode chưa nằm trong map
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticon
    "\U0001F300-\U0001F5FF"   # symbols & pictograph
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"   # supplemental
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002B50"
    "]+"
)

def convert_emoji(text: str) -> str:
    """Chuyển emoji/icon thành token cảm xúc tiếng Việt."""
    for emoji_char, token in EMOJI_MAP.items():
        text = text.replace(emoji_char, token)
    # Xoá emoji còn sót (không nằm trong map)
    text = _EMOJI_RE.sub(" [EMOJI] ", text)
    return text

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:
    """Tiền xử lý tối thiểu, giữ đặc trưng ngôn ngữ."""
    if pd.isna(text):
        return ""
    text = str(text)

    # 1. Loại HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Loại URLs
    text = re.sub(r"http[s]?://\S+", " [URL] ", text)
    text = re.sub(r"www\.\S+", " [URL] ", text)

    # 3. Loại email
    text = re.sub(r"\S+@\S+", " [EMAIL] ", text)

    # 4. Loại số điện thoại
    text = re.sub(r"(\+84|0)\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}", " [PHONE] ", text)

    # 5. Convert emoji → token cảm xúc
    text = convert_emoji(text)

    # 6. Chuẩn hoá khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text

def is_valid(text: str) -> bool:
    return isinstance(text, str) and len(text.split()) >= 5

# ============================================================
# LOAD & SPLIT DATA
# ============================================================

def load_and_split(config: dict):
    print("\n" + "=" * 80)
    print("📂 LOADING DATA")
    print("=" * 80)

    data_path = config["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Raw rows: {len(df)}")

    # Rename columns
    if "post_message" in df.columns:
        df.rename(columns={"post_message": "text"}, inplace=True)
    if "text" not in df.columns:
        raise ValueError("❌ Cannot find text column (post_message)")
    if "label" not in df.columns:
        raise ValueError("❌ Cannot find label column")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Validate labels
    bad = df[~df["label"].isin([0, 1])]
    if len(bad) > 0:
        raise ValueError(f"❌ Found labels not in {{0,1}}. Examples:\n{bad.head()}")

    # Clean text
    print("🧹 Cleaning text + converting emoji → emotion tokens ...")
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].apply(is_valid)].copy()

    # Dedup
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean"], keep="first").reset_index(drop=True)
    print(f"✅ After clean + dedup: {len(df)} (removed {before - len(df)} duplicates)")

    n = len(df)
    c0 = int((df["label"] == 0).sum())
    c1 = int((df["label"] == 1).sum())
    print(f"✅ Total: {n} | Reliable(0)={c0} ({c0/n:.1%}) | Unreliable(1)={c1} ({c1/n:.1%})")

    # ----- Stratified split: 80 / 10 / 10 -----
    print("\n" + "=" * 80)
    print("✂️  SPLITTING DATA: 80% Train / 10% Val / 10% Test (stratified)")
    print("=" * 80)

    seed = config["seed"]
    test_size = config["test_size"]
    val_size = config["val_size"]

    # Step 1: tách 80% train  +  20% temp
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size),
        stratify=df["label"], random_state=seed,
    )

    # Step 2: tách temp 50/50 → val + test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["label"], random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    def _print_split(name, d):
        n = len(d)
        c0 = int((d["label"] == 0).sum())
        c1 = int((d["label"] == 1).sum())
        print(f"  {name:6s}: {n:>6d} | Reliable={c0} ({c0/n:.1%}) | Unreliable={c1} ({c1/n:.1%})")

    _print_split("Train", train_df)
    _print_split("Val", val_df)
    _print_split("Test", test_df)

    return train_df, val_df, test_df

# ============================================================
# DATASET
# ============================================================

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

# ============================================================
# HELPERS
# ============================================================

def compute_class_weights(labels, device):
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    return counts, torch.tensor(weights, dtype=torch.float32, device=device)


def freeze_backbone(model, freeze_layers: int, freeze_embeddings: bool):
    if freeze_layers <= 0 and not freeze_embeddings:
        return
    if hasattr(model, "electra"):
        if freeze_embeddings and hasattr(model.electra, "embeddings"):
            for p in model.electra.embeddings.parameters():
                p.requires_grad = False
        if hasattr(model.electra, "encoder"):
            for i, layer in enumerate(model.electra.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧊 Frozen: {total - trainable:,} / {total:,} params  |  Trainable: {trainable:,}")


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, labels)
        total_loss += loss.item()
        preds = out.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1m, np.array(all_labels), np.array(all_preds)

# ============================================================
# TRAIN
# ============================================================

def train(config: dict):
    device = config["device"]

    # --- data ---
    train_df, val_df, test_df = load_and_split(config)

    # --- tokenizer ---
    print("\n" + "=" * 80)
    print(f"🤖 MODEL: {config['model_name']}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False)

    train_loader = DataLoader(
        NewsDataset(train_df["text_clean"].values, train_df["label"].values,
                    tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        NewsDataset(val_df["text_clean"].values, val_df["label"].values,
                    tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=False,
    )
    test_loader = DataLoader(
        NewsDataset(test_df["text_clean"].values, test_df["label"].values,
                    tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=False,
    )

    # --- model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_labels"],
    ).to(device)

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # dropout
    try:
        model.config.hidden_dropout_prob = config["dropout"]
        model.config.attention_probs_dropout_prob = config["dropout"]
    except Exception:
        pass

    freeze_backbone(model, config["freeze_layers"], config["freeze_embeddings"])

    # --- class weights ---
    counts, class_w = compute_class_weights(
        train_df["label"].values.astype(int), device,
    )
    print(f"📊 Class counts [Reliable, Unreliable]: {counts}")
    print(f"   Class weights: {class_w.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # --- optimizer & scheduler ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # --- training loop ---
    print("\n" + "=" * 80)
    print("🚀 TRAINING")
    print("=" * 80)
    print(f"   Device     : {device}")
    print(f"   Epochs     : {config['epochs']}")
    print(f"   Batch size : {config['batch_size']}")
    print(f"   LR         : {config['learning_rate']}")
    print(f"   Max length : {config['max_length']}")
    print(f"   Total steps: {total_steps}  (warmup: {warmup_steps})")
    print()

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=True)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)

        # --- validation ---
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            marker = " ✅ best"

        print(
            f"  → train_loss={train_loss:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
            f"val_macroF1={val_f1:.4f}{marker}"
        )

    # --- load best ---
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n🏆 Best Val Macro-F1: {best_f1:.4f}")

    # ============================================================
    # EVALUATION ON TEST SET
    # ============================================================

    print("\n" + "=" * 80)
    print("🎯 FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)

    f1_reliable = f1_score(y_true, y_pred, pos_label=0, average="binary")
    f1_unreliable = f1_score(y_true, y_pred, pos_label=1, average="binary")

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Macro-F1      : {test_f1:.4f}")
    print(f"  F1 Reliable(0): {f1_reliable:.4f}")
    print(f"  F1 Unreliable(1): {f1_unreliable:.4f}")
    print(f"  Gap           : {abs(f1_reliable - f1_unreliable):.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["Reliable (0)", "Unreliable (1)"],
        digits=4,
    ))

    # ============================================================
    # SAVE MODEL
    # ============================================================

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("=" * 80)
    print(f"💾 Model saved to: {output_dir}")
    print("=" * 80)

    return model, tokenizer

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train(CONFIG)
