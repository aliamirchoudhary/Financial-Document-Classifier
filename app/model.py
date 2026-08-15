"""Model loading + classification for the service.

Loads a fine-tuned DistilBERT checkpoint from a directory containing
config.json / model.safetensors / tokenizer.json (plus optional vocab.txt).
The saved config stores generic LABEL_0..16, so the real 17-class label map is
rebuilt from the training corpus (sorted unique labels == training order), with
a hardcoded fallback if the corpus is absent at deploy time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

FALLBACK_LABELS = sorted({
    "Articles of Incorporation - Certificate of Formation",
    "Audit Report",
    "Balance Sheet- Statement of Financial Position",
    "Bank Reconciliation",
    "Bank Statements- Cr Card Statements",
    "Board Minutes - Resolutions",
    "Cash Flow Statement- Statement of Cashflow",
    "Engagement Letter",
    "Fund Statement",
    "General Ledger",
    "Invoices",
    "Loan Details- Lease Details",
    "Payroll Register- Payroll Slips",
    "Payroll Summary",
    "Statement of Activities- Statement of Profit & Loss- Income Statement",
    "Trial Balance",
    "Unclassified",
})


class Classifier:
    def __init__(self, model_dir: str | Path, corpus_jsonl: str | Path | None = None):
        self.model_dir = Path(model_dir)
        self.labels: list[str] = list(FALLBACK_LABELS)
        self.id2label: dict[int, str] = {i: lab for i, lab in enumerate(self.labels)}

        if corpus_jsonl is not None and Path(corpus_jsonl).exists():
            labels: set[str] = set()
            with open(corpus_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and '"label"' in line:
                        labels.add(json.loads(line)["label"])
            if labels:
                self.labels = sorted(labels)
                self.id2label = {i: lab for i, lab in enumerate(self.labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            num_labels=len(self.id2label),
            id2label={str(i): lab for i, lab in self.id2label.items()},
            label2id={lab: i for i, lab in self.id2label.items()},
        )
        self.model.eval()

    def classes(self) -> list[str]:
        return self.labels

    @torch.inference_mode()
    def classify_text(self, text: str, truncate_chars: int = 3000,
                      max_len: int = 512, top_k: int = 5):
        enc = self.tokenizer(
            (text or "")[:truncate_chars], padding=True, truncation=True,
            max_length=max_len, return_tensors="pt")
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        order = np.argsort(-probs)
        top_id = int(order[0])
        return {
            "predicted_class": self.id2label[top_id],
            "class_id": top_id,
            "confidence": float(probs[top_id]),
            "num_classes": len(self.id2label),
            "top_k": [
                {"class": self.id2label[int(i)], "id": int(i), "probability": float(probs[i])}
                for i in order[:top_k]
            ],
        }