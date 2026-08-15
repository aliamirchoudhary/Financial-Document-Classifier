# Model weights — handled via Git LFS

The fine-tuned checkpoint **is versioned in this repository via Git LFS**:
the 255 MB checkpoint would otherwise bloat every clone, and Git LFS keeps it
pull-able without a special artifact server.

## What is tracked (see `.gitattributes`)

| File | Size | Git LFS? |
|---|---:|---|
| `model.safetensors` | ~255 MB | ✅ |
| `tokenizer.json` | ~0.7 MB | ✅ |
| `tokenizer_config.json` | <1 KB | ✅ |
| `config.json` | <1 KB | ✅ |

> Keeping the full checkpoint set under LFS makes reloads **atomic** — a
> checkout is never left half-LFS / half-plain.

## Workflow (LFS-aware)

| Action | Command |
|---|---|
| Clone (pulls pointers) | `git clone <url>` + `git lfs pull` |
| Update checkpoint | `git lfs ls-files -l` to confirm the local copy vs. the pointer |
| Verify integrity (after pull) | `Get-FileHash -Algorithm SHA256 models\model.safetensors` |
| Redownload a file | `git lfs fetch --all; git lfs checkout` |

## Official SHA-256

| File | SHA-256 |
|---|---|
| `model.safetensors` | *(compute after first LFS pull; keep table updated on each refresh)* |

> A mismatch means a corrupted/unauthorized copy — do not run it.

## Provenance

- Model: `distilbert-base-uncased`, fine-tuned, 17 labels
- Training: see [`docs/pipeline.md`](../docs/pipeline.md)
- Reported: 0.945 acc / 0.933 macro-F1 / 0.945 weighted-F1 (true held-out)