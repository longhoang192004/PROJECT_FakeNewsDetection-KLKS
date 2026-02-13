# PipelineV2 — Unified Data Processing & Training

Pipeline xu ly du lieu va training model cho **Vietnamese Fake News Detection**.

## Cau truc thu muc

```
PipelineV2/
├── config.py          # Cau hinh tap trung (paths, thresholds, model names)
├── clean_text.py      # Lam sach text: Unicode NFC, emoji, URL/email/phone
├── dedup.py           # Phat hien near-duplicate (TF-IDF + Union-Find)
├── split.py           # Chia data chong leak (StratifiedGroupKFold)
├── dataset.py         # Dataset + DataLoader (dynamic padding, augmentation)
├── pipeline.py        # Orchestrator: load → clean → dedup → split → save
├── run_pipeline.py    # CLI chay data pipeline
└── train_phobert.py   # Fine-tune PhoBERT (vinai/phobert-base)
```

## Thu tu chay

### Buoc 1: Cai dat dependencies

```bash
pip install -r requirements.txt
```

### Buoc 2: Chay data pipeline (tao train/val/test)

```bash
cd CodeKLKS
python -m PipelineV2.run_pipeline --data_path ../fakenewsdatasetv1.csv
```

Thu tu xu ly ben trong:
1. `config.py` — load cau hinh
2. `clean_text.py` — lam sach text (Unicode NFC, emoji, URL/email/phone)
3. `dedup.py` — phat hien va gom nhom near-duplicate
4. `split.py` — chia train/val/test chong leak (StratifiedGroupKFold)

**Output** (luu vao `outputs/pipeline_v2/`):

| File | Mo ta |
|------|-------|
| `train.csv` | Tap train (~80%) |
| `val.csv` | Tap validation (~10%) |
| `test.csv` | Tap test (~10%) |
| `dataset_clean_dedup.csv` | Toan bo data da clean + dedup |
| `leak_report.json` | Bao cao leak (exact + near-dup) |
| `stats.json` | Thong ke dataset |

### Buoc 3: Train PhoBERT

```bash
cd CodeKLKS
python -m PipelineV2.train_phobert
```

> **Luu y:** Buoc 3 se tu dong chay lai Buoc 2 truoc khi train.
> Neu da co data tu Buoc 2, van chay lai de dam bao nhat quan.

Tuy chinh:

| Flag | Default | Mo ta |
|------|---------|-------|
| `--epochs` | 5 | So epoch |
| `--batch_size` | 8 | Batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max token length |
| `--freeze_layers` | 0 | Dong bang N layer dau |
| `--freeze_embeddings` | False | Dong bang embedding |
| `--augment_prob` | 0.3 | Xac suat augment |

### 3. Import trong code khac

```python
from PipelineV2.config import get_config
from PipelineV2.pipeline import run_pipeline
from PipelineV2.dataset import make_single_loaders, compute_class_weights

config = get_config()
splits = run_pipeline(config)
```

## Cai tien so voi code cu

- **Clean text thong nhat** — 1 ham duy nhat thay vi moi script 1 phien ban khac
- **Chong data leakage** — near-dup groups giu nguyen khi split
- **Dynamic padding** — pad theo batch thay vi global max_length, tiet kiem GPU
- **Text augmentation** — random word delete/swap khi train
- **Class weights** — xu ly imbalanced data tu dong
