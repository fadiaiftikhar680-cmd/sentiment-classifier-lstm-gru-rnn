"""
Sentiment Analysis Backend API
==============================
FastAPI backend that serves the trained RNN / LSTM / GRU sentiment
analysis models produced by `Sentiment_Analysis_RNN_LSTM_GRU.ipynb`.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import re
import json
import pickle
import string
import time
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# Paths (relative to this file, so it works no matter where uvicorn is run)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIG_PATH = os.path.join(TOKENIZER_DIR, "config.json")
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.pickle")
LABEL_ENCODER_PATH = os.path.join(TOKENIZER_DIR, "label_encoder.pickle")
COMPARISON_CSV_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

# --------------------------------------------------------------------------
# NLTK resources (safe to call every start — no-ops if already downloaded)
# --------------------------------------------------------------------------
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Exact same preprocessing used during training — must stay identical
    to the notebook's `clean_text` function or predictions will be wrong."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Globals populated on startup
# --------------------------------------------------------------------------
CONFIG: dict = {}
TOKENIZER = None
LABEL_ENCODER = None
MODELS: dict = {}          # name -> loaded keras model
DEFAULT_MODEL_NAME: Optional[str] = None
IDX_TO_LABEL: dict = {}    # {0: "negative", 1: "positive"}

app = FastAPI(
    title="Sentiment Analysis API",
    description="Serves RNN / LSTM / GRU models trained on the IMDb 50K review dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_artifacts():
    global CONFIG, TOKENIZER, LABEL_ENCODER, MODELS, DEFAULT_MODEL_NAME, IDX_TO_LABEL

    if not os.path.exists(CONFIG_PATH):
        print(f"[WARN] config.json not found at {CONFIG_PATH}. "
              f"Copy your notebook's tokenizer/ and models/ folders into backend/ first.")
        return

    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)

    with open(TOKENIZER_PATH, "rb") as f:
        TOKENIZER = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        LABEL_ENCODER = pickle.load(f)

    IDX_TO_LABEL = {v: k for k, v in CONFIG.get("label_mapping", {}).items()}

    # Load every available model referenced in config.json
    for name, rel_path in CONFIG.get("available_models", {}).items():
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            print(f"Loading {name} model from {full_path} ...")
            MODELS[name] = tf.keras.models.load_model(full_path)
        else:
            print(f"[WARN] {name} model missing at {full_path}, skipping.")

    DEFAULT_MODEL_NAME = CONFIG.get("best_model_name")
    if DEFAULT_MODEL_NAME not in MODELS and MODELS:
        DEFAULT_MODEL_NAME = list(MODELS.keys())[0]

    print(f"Startup complete. Loaded models: {list(MODELS.keys())}. "
          f"Default/best model: {DEFAULT_MODEL_NAME}")


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw movie review text")
    model: Optional[str] = Field(
        None, description="Which model to use: RNN, LSTM, or GRU. Defaults to the best model."
    )


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float
    model_used: str
    inference_time_ms: float
    cleaned_text: str


class ModelInfo(BaseModel):
    name: str
    loaded: bool


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running.",
        "docs": "/docs",
        "models_loaded": list(MODELS.keys()),
        "default_model": DEFAULT_MODEL_NAME,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODELS else "artifacts_missing",
        "models_loaded": list(MODELS.keys()),
    }


@app.get("/model-info")
def model_info():
    if not CONFIG:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded on the server.")
    return {
        "config": CONFIG,
        "models_available": list(MODELS.keys()),
        "default_model": DEFAULT_MODEL_NAME,
    }


@app.get("/model-comparison")
def model_comparison():
    """Returns the RNN vs LSTM vs GRU comparison table (for frontend charts)."""
    if not os.path.exists(COMPARISON_CSV_PATH):
        raise HTTPException(
            status_code=404,
            detail="model_comparison.csv not found. Copy results/ folder from the notebook.",
        )
    df = pd.read_csv(COMPARISON_CSV_PATH)
    return json.loads(df.to_json(orient="records"))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Copy models/ and tokenizer/ from the notebook into backend/.",
        )

    model_name = (payload.model or DEFAULT_MODEL_NAME or "").upper()
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose one of {list(MODELS.keys())}.",
        )

    model = MODELS[model_name]

    start = time.time()
    cleaned = clean_text(payload.text)
    seq = TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=CONFIG["max_len"],
        padding="post",
        truncating="post",
    )

    prob_positive = float(model.predict(padded, verbose=0)[0][0])
    prob_negative = 1.0 - prob_positive
    elapsed_ms = (time.time() - start) * 1000

    predicted_idx = 1 if prob_positive >= 0.5 else 0
    sentiment_label = IDX_TO_LABEL.get(predicted_idx, "positive" if predicted_idx == 1 else "negative")
    confidence = prob_positive if predicted_idx == 1 else prob_negative

    return PredictResponse(
        sentiment=sentiment_label,
        confidence=round(confidence, 4),
        probability_positive=round(prob_positive, 4),
        probability_negative=round(prob_negative, 4),
        model_used=model_name,
        inference_time_ms=round(elapsed_ms, 2),
        cleaned_text=cleaned,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
