"""FastAPI service for the Pipeline-2 document classifier.

Endpoints:
  GET  /health          alive + model/class info (no auth)
  GET  /classes         the 17 class names
  POST /predict         multipart file upload -> prediction
  POST /predict_text    raw text in JSON -> prediction

Security posture (for confidential financial docs):
  - optional Bearer token auth (set AUTH_TOKEN env; if unset, auth is off)
  - request bodies / extracted text are NEVER logged
  - stateless: nothing is persisted, no database, no disk writes beyond temp
  Deploy the checkpoint files as a volume / baked into the image.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.extract import extract_text_bytes
from app.model import Classifier

# ---------------------------------------------------------------- config

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models"))
CORPUS = os.environ.get("CORPUS_JSONL", None)   # optional: rebuild label map from corpus
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "").strip()
MAX_NUM_PAGES = int(os.environ.get("MAX_NUM_PAGES", "4"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# ---------------------------------------------------------------- app

app = FastAPI(title="Financial Audit Document Classifier",
              version="1.0.0",
              docs_url="/docs", redoc_url=None)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

try:
    _cls = Classifier(MODEL_DIR, corpus_jsonl=CORPUS if CORPUS else None)
    print(f"model loaded from {MODEL_DIR} | {len(_cls.classes())} classes", flush=True)
except Exception as e:  # defer error until /health so the app still reports it
    print(f"MODEL LOAD FAILED: {type(e).__name__}: {e}", flush=True)
    _cls = None
    _load_error = str(e)
else:
    _load_error = None


# ---------------------------------------------------------------- auth

def require_token(authorization: str | None = Header(None)):
    """If AUTH_TOKEN is set, requests need `Authorization: Bearer <token>`."""
    if not AUTH_TOKEN:
        return
    if authorization is None or authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ---------------------------------------------------------------- schemas

class PredictTextRequest(BaseModel):
    text: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int | None = None
    classes: list[str] | None = None
    error: str | None = None


# ---------------------------------------------------------------- endpoints

@app.get("/health", response_model=Health)
def health():
    if _cls is None:
        return Health(status="error", model_loaded=False, error=_load_error)
    return Health(status="ok", model_loaded=True,
                  num_classes=len(_cls.classes()), classes=_cls.classes())


@app.get("/classes")
def classes(dep: None = Depends(require_token)):
    if _cls is None:
        raise HTTPException(status_code=500, detail=f"model not loaded: {_load_error}")
    return {"classes": _cls.classes(), "num_classes": len(_cls.classes())}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  auth: None = Depends(require_token)):
    if _cls is None:
        raise HTTPException(status_code=500, detail=f"model not loaded: {_load_error}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file upload")
    safe_name = Path(file.filename or "upload").name
    pages = extract_text_bytes(safe_name, data, max_pages=MAX_NUM_PAGES)
    page_texts = {str(p["page_num"]): p["text"] for p in pages}
    page1 = page_texts.get("1", "")
    full = "\n".join(page_texts.values()).strip()
    text = page1 if page1 else full
    result = _cls.classify_text(text, top_k=TOP_K)
    # NOTE: intentionally do NOT return the extracted text (confidential diff)
    return {
        "filename": safe_name,
        "pages_extracted": len(pages),
        "text_chars": len(text),
        "using": "page1" if page1 else "full_document",
        **result,
    }


@app.post("/predict_text")
async def predict_text(req: PredictTextRequest,
                       auth: None = Depends(require_token)):
    if _cls is None:
        raise HTTPException(status_code=500, detail=f"model not loaded: {_load_error}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    return _cls.classify_text(req.text, top_k=TOP_K)