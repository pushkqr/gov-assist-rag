"""IndicTrans2 translation service.

Wraps ai4bharat/indictrans2 behind the contract the application already speaks:

    POST /translate  {"text": "...", "src_lang": "mar_Deva", "tgt_lang": "eng_Latn"}
    ->               {"translated_text": "..."}

Language codes are FLORES-200 tags, which is what IndicTrans2 expects natively.
"""

import os
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = os.getenv("INDICTRANS_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = int(os.getenv("TRANSLATE_MAX_TOKENS", "256"))

app = FastAPI(title="Mimir Translation")

_tokenizer = None
_model = None
_processor = None
# Generation is not thread-safe across concurrent requests on a single model instance,
# and uvicorn will happily run handlers concurrently.
_lock = threading.Lock()


def _load():
    global _tokenizer, _model, _processor
    if _model is not None:
        return
    from IndicTransToolkit.processor import IndicProcessor

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    _model.eval()
    _processor = IndicProcessor(inference=True)


@app.on_event("startup")
def startup():
    # Loaded at boot rather than on first request: the first translation would otherwise
    # absorb the whole model load and blow the query latency budget.
    _load()


class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "mar_Deva"
    tgt_lang: str = "eng_Latn"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/translate")
def translate(req: TranslateRequest):
    text = (req.text or "").strip()
    if not text:
        return {"translated_text": ""}

    with _lock:
        batch = _processor.preprocess_batch([text], src_lang=req.src_lang, tgt_lang=req.tgt_lang)
        encoded = _tokenizer(batch, truncation=True, padding="longest",
                             return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = _model.generate(**encoded, max_length=MAX_TOKENS, num_beams=5,
                                        num_return_sequences=1)
        decoded = _tokenizer.batch_decode(generated, skip_special_tokens=True)
        output = _processor.postprocess_batch(decoded, lang=req.tgt_lang)

    return {"translated_text": output[0] if output else text}
