# Model Pipeline — how the classifier was built

This documents where the model came from and how it was trained. The service
itself (app/) is reproducible from a few artifacts; this page explains the
provenance and experiment history (notebooks 00–06).

## The problem we set out to fix

The previous classifier used **LayoutLMv3** (layout + vision + text). It scored
**~91% overall** but only **3/17 classes** at 100% per-class — the per-class
picture was weak, and it needed heavy vision/OCR infrastructure. We wanted a
**text-only** model that maximizes **per-class** (not just overall) accuracy, on
a 16-known + `Unclassified` taxonomy. 17 of 49 total classes are banked in this
repo; the same playbook generalizes to the rest.

## Data

- **Source:** 17 labeled folders, 2,295 files.
- **Final corpus:** 2,290 clean records (all formats) — see table.

| Format | Records | % |
|---|---|---|
| PDF (native + scanned) | 1,864 | 81% |
| XLSX | 321 | 14% |
| DOCX | 51 | 2% |
| CSV | 38 | 2% |
| XLS | 7 | <1% |
| XLSM | 6 | <1% |
| JFIF / PNG | 1 + 1 | <1% |

Extraction is rotation-aware and quality-gated (`app/extract.py`): cheap
reading-order pass first, multiword readability metric detects rotated/garbage
OCR, then a 4-rotation OCR fallback so upside-down scans still extract.

## Experiment history (Colab, notebooks 00–06)

| # | Notebook | Question asked | Finding |
|---|---|---|---|
| 00 | setup | Is the corpus sane? | Confirmed imbalance (Unclassified heavy) |
| 01 | learning curve | Does more data help? | Yes — sharp gains per class |
| 02 | page-signal | Does long context help? | **No — page 1 ≈ full doc.** Chose `page1` |
| 03 | bake-off | Best model? | DistilBERT > ModernBERT > DeBERTa-v3* |
| 04 | weighted | Does class-weighting + margin help? | sqrt weighting + a margin *de-escape* rule (circular sweep flagged) |
| 05 | final | Honest score? | **0.945 acc / 0.933 macro / 0.945 wtd** on true held-out |
| 06 | reload | Is the saved model usable? | Yes — loads & classifies real docs |

*\* DeBERTa-v3 collapsed to ~0.02 accuracy across fp16 and fp32 with grad-clip;
abandoned as likely training instability, time-boxed — not proven inferior.*

## Final configuration

- **Model:** `distilbert-base-uncased`, sequence classification, 17 labels
- **Window:** page 1 only (reading-order text)
- **Loss:** weighted CrossEntropy, **sqrt** inverse-frequency weights
- **Margin rule:** Unclassified → runner-up when winning edge ≤ 0.16, targets
  restricted to the narrow set (Loan Details, Articles of Incorporation)
- **Validation:** 80/20 stratified holdout, seed 42. Margin + weights tuned on
  inner-CV of the 80% **only**; applied once to the held-out 20%.

## Reported per-class F1 (true held-out)

Cash Flow 1.000 · Payroll Summary 1.000 · Bank Statements 0.992 ·
Bank Reconciliation 0.982 · Engagement Letter 0.970 · Invoices 0.966 ·
Loan Details 0.947 · Balance Sheet 0.938 · Board Minutes 0.929 ·
Unclassified 0.929 · Fund Statement 0.914 · Statement of Activities 0.914 ·
Articles of Incorporation 0.900 · Payroll Register 0.884 · General Ledger 0.870 ·
Trial Balance 0.870 · **Audit Report 0.857** (softest)

## Honest caveats

- `Unclassified` is a trained positive class — never force-remapped; the margin
  rule only de-escapes it as a review aid.
- Audit Report / General Ledger / Trial Balance are the softest (mutually
  similar, sparsely populated). Margin=0.16 changed 0 samples on this holdout;
  its benefit is inner-CV-validated, unproven on this tail.
- Layout/OCR intentionally excluded — page-1 text must exist and be legible.
- Low-confidence predictions (≈0.5) should be human-reviewed.