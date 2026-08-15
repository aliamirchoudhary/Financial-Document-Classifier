<div align="center">

<img src="https://img.shields.io/badge/status-active-success" alt="Status"/>&nbsp;
<img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI"/>&nbsp;
<img src="https://img.shields.io/badge/model-DistilBERT-FF6F00" alt="DistilBERT"/>&nbsp;
<img src="https://img.shields.io/badge/accuracy-94.5%25-brightgreen" alt="Accuracy"/>&nbsp;
<img src="https://img.shields.io/badge/macro%20F1-93.3%25-28a745" alt="Macro F1"/>&nbsp;
<img src="https://img.shields.io/badge/weights-Git%20LFS-2ea44f" alt="Weights via Git LFS"/>&nbsp;
<img src="https://img.shields.io/badge/visibility-public-1f6feb" alt="Public"/>&nbsp;
<img src="https://img.shields.io/badge/license-MIT-555" alt="MIT License"/>

# Financial Audit Document Classifier

**A text-only document classifier for financial-audit document sets — predicting 17 document classes from raw file uploads.**

Accepts **PDF (native & scanned), DOCX, CSV, XLS/XLSX/XLSM, PNG/JPG/JFIF** and returns a class prediction with calibrated-confidence scores.

</div>

---

## ✨ At a glance

| What | Why it matters |
|---|---|
| 📄 **17-class taxonomy** | Balance Sheet, General Ledger, Bank Reconciliation, Trial Balance, Invoices, … plus a trained `Unclassified` catch-all |
| 🧠 **Fine-tuned DistilBERT, text-only** | No heavy layout/vision stack — runs on a single CPU core |
| 🎯 **Per-class accuracy obsessed** | max 100% min 85.7% on every class, not just overall |
| 🔒 **Confidential-data friendly** | stateless, token-auth, zero body logging |
| 🔌 **Self-contained** | extraction → model → API in one small package |

> **Validated honestly.** The headline numbers come from a **true held-out 20%** that was quarantined during tuning — the model never saw it while choosing hyper-parameters.

| Metric | Score |
|:---|---:|
| **Accuracy** | **0.945** |
| **Macro-F1** | **0.933** |
| **Weighted-F1** | **0.945** |
| Min per-class F1 | 0.857 (Audit Report) |

All **17 classes ≥ 0.857**, with **Cash Flow** and **Payroll Summary** at **1.000**.

---

## 🔍 How it works

```
               ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
               │     EXTRACT         │      │       MODEL         │      │        API          │
   file ─────▶ │   app/extract.py   ─┼────▶ │   app/model.py     ─┼────▶ │    app/main.py     ─┼─▶ JSON
  upload      └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
               reading-order text           fine-tuned DistilBERT        FastAPI endpoints
               PDF/OCR/tabular/docx         page-1 signal + sqrt-weights  /predict /predict_text
```

Three focused modules — no hidden machinery.

### 1️⃣ Extract — turn any file into text
- **PDF** — cheap reading-order pass; a *multiword-readability metric* spots rotated/garbage OCR and triggers a **4-rotation OCR** fallback so upside-down scans still parse.
- **Spreadsheets** (CSV/XLS/XLSX/XLSM) — flattened to tabular text rows.
- **DOCX** — paragraphs + table cells.
- **Images** (PNG/JPG/JFIF) — routed through the OCR path.

### 2️⃣ Model — score the text
- Fine-tuned **`distilbert-base-uncased`** sequence classifier, **17 labels**.
- **Page-1 window** (research showed page-1 ≈ full-document signal for these docs).
- **sqrt inverse-frequency weighting** elevates the data-starved classes.
- Auto-rebuilds the real 17-class label map (the saved config only stores `LABEL_0..16`).

### 3️⃣ API — expose it cleanly
- **FastAPI** with auto-generated Swagger UI at `/docs`.
- **Bearer-token auth** (optional), **no request-body logging**, fully **stateless** — built for confidential financial data.

> Full model provenance, the 00–06 experiment trail, and honest caveats: **[`docs/pipeline.md`](docs/pipeline.md)**.

---

## 🚀 Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Point at a checkpoint (config.json + model.safetensors + tokenizer.json)
export MODEL_DIR=/path/to/checkpoint            # Windows: set MODEL_DIR=C:\path\to\checkpoint
export AUTH_TOKEN=your-secret-token             # optional: protect /predict & /classes
# export CORPUS_JSONL=/path/to/colab.jsonl      # optional: rebuild class map from corpus

# 3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows? Just run **`run_local.bat`** (pre-wired to a local checkpoint + corpus).

Then open **<http://localhost:8000/docs>** → `/predict` → **Try it out** → upload a file → **Execute**.

### 🧪 Try it from the terminal

```bash
# Health + class list
curl  http://localhost:8000/health
curl -H "Authorization: Bearer your-secret-token" http://localhost:8000/classes

# Upload a document
curl -X POST http://localhost:8000/predict \
     -H "Authorization: Bearer your-secret-token" \
     -F "file=@some/report.pdf"

# Raw text
curl -X POST http://localhost:8000/predict_text \
     -H "Authorization: Bearer your-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"text":"Trial balance as of Dec 31, 2024 ..."}'
```

---

## 📡 API reference

| Method | Path | Auth | Description |
|:---:|:---|:---:|---|
| `GET` | `/health` | — | Service + model status, class count |
| `GET` | `/classes` | 🔐 | The 17 class names |
| `POST` | `/predict` | 🔐 | Multipart **file upload** → prediction |
| `POST` | `/predict_text` | 🔐 | Raw text (JSON) → prediction |
| `GET` | `/docs` | — | Interactive Swagger UI |

**Example response** (`/predict`):

```json
{
  "filename": "report.pdf",
  "pages_extracted": 3,
  "text_chars": 542,
  "using": "page1",
  "predicted_class": "Trial Balance",
  "class_id": 15,
  "confidence": 0.4353,
  "num_classes": 17,
  "top_k": [
    {"class": "Trial Balance", "id": 15, "probability": 0.4353},
    {"class": "Balance Sheet - Statement of Financial Position", "id": 2, "probability": 0.1600}
  ]
}
```

> `confidence` ∈ [0,1] — how decisively the top class beat its rivals; `top_k`
> shows why. Low-confidence outputs (≈0.5) are worth a human double-check.

---

## 🔒 Security & privacy posture

| Property | Design |
|---|---|
| **Stateless** | No database, no persistence — the only disk write is a temp file, deleted after use |
| **No body logging** | Document text is never written to logs |
| **Token auth** | `AUTH_TOKEN` required on `/classes`, `/predict`, `/predict_text`; 401 otherwise |
| **Weights management** | Checkpoint tracked via **Git LFS** (pulled with clone; see [`models/README.md`](models/README.md)) |
| **Provenance** | [`docs/pipeline.md`](docs/pipeline.md) records how the model was trained & scored |

---

## 🗂 Repo layout

```
.
├── app/                    # the service
│   ├── main.py             #   FastAPI endpoints + auth
│   ├── model.py            #   checkpoint load + label map + classify
│   └── extract.py          #   PDF/OCR, tabular, docx, image text extraction
├── models/                 # ← checkpoint via Git LFS (see models/README.md)
├── data/                   # optional training corpus / artifacts (never committed)
├── docs/
│   └── pipeline.md         # model provenance + experiment history (00–06)
├── tests/
│   └── test_api.py         # API smoke tests (pytest)
├── requirements.txt        # pinned deps
├── .env.example            # config template
├── run_local.bat           # one-click local runner (Windows)
└── README.md
```

---

## 🏷 The 17 classes

```
Articles of Incorporation · Audit Report · Balance Sheet · Bank Reconciliation ·
Bank Statements · Board Minutes · Cash Flow Statement · Engagement Letter ·
Fund Statement · General Ledger · Invoices · Loan Details · Payroll Register ·
Payroll Summary · Statement of Activities / P&L · Trial Balance · Unclassified
```

Scored per-class F1:

| Class | F1 | | Class | F1 |
|---|---:|---|---|---:|
| Cash Flow Statement | **1.000** | | Fund Statement | 0.914 |
| Payroll Summary | **1.000** | | Statement of Activities / P&L | 0.914 |
| Bank Statements / CC | 0.992 | | Articles of Incorporation | 0.900 |
| Bank Reconciliation | 0.982 | | Payroll Register / Slips | 0.884 |
| Engagement Letter | 0.970 | | General Ledger | 0.870 |
| Invoices | 0.966 | | Trial Balance | 0.870 |
| Loan Details / Lease | 0.947 | | **Audit Report** | **0.857** |
| Balance Sheet / SFP | 0.938 | | | |
| Board Minutes | 0.929 | | | |
| Unclassified | 0.929 | | | |

---

## 🧪 Tests

```bash
export MODEL_DIR=... AUTH_TOKEN=test
python -m pytest tests/test_api.py
```

---

###### License: [MIT](LICENSE). Training data and source documents are NOT redistributed in this repo.