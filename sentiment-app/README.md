# CineSense — Sentiment Analysis Web App

Backend (FastAPI) + Frontend (HTML/CSS/JS) for the RNN / LSTM / GRU sentiment
models trained in `Sentiment_Analysis_RNN_LSTM_GRU.ipynb`.

```
sentiment-app/
├── backend/
│   ├── main.py                # FastAPI app — loads models, exposes /predict
│   ├── requirements.txt
│   ├── models/                # ⚠️ put your trained .keras files here
│   ├── tokenizer/              # ⚠️ put tokenizer.pickle, label_encoder.pickle, config.json here
│   └── results/                # ⚠️ put model_comparison.csv here (optional, for charts)
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Step 1 — Copy your trained artifacts from the notebook

After running the notebook end-to-end, it saves these files (Part 10 & Part 15):

| From notebook folder | Copy into |
|---|---|
| `models/rnn.keras` | `backend/models/rnn.keras` |
| `models/lstm.keras` | `backend/models/lstm.keras` |
| `models/gru.keras` | `backend/models/gru.keras` |
| `models/best_model.keras` | `backend/models/best_model.keras` |
| `tokenizer/tokenizer.pickle` | `backend/tokenizer/tokenizer.pickle` |
| `tokenizer/label_encoder.pickle` | `backend/tokenizer/label_encoder.pickle` |
| `tokenizer/config.json` | `backend/tokenizer/config.json` |
| `results/model_comparison.csv` | `backend/results/model_comparison.csv` |

If the notebook ran in Google Colab, download these folders (zip them first)
and drag them into the matching `backend/` subfolders in VS Code.

## Step 2 — Run the backend

Open a terminal in `backend/`:

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/docs` — you should see the
Swagger UI with `/predict`, `/health`, `/model-comparison` endpoints.

## Step 3 — Run the frontend

The frontend is plain HTML/CSS/JS — no build step. Easiest options in VS Code:

- Install the **Live Server** extension → right-click `frontend/index.html` → "Open with Live Server"
- Or just double-click `frontend/index.html` to open it in your browser

The frontend calls `http://localhost:8000` by default (see `API_BASE` at the
top of `script.js`) — change that if your backend runs elsewhere.

## What you get

- A dark, cinema-marquee themed UI (gold/crimson palette, film-strip motif)
- Live sentiment prediction with an animated confidence gauge + probability bars
- A model picker (RNN / LSTM / GRU / auto-best)
- Comparison charts (accuracy, F1, training time) pulled straight from
  `results/model_comparison.csv`
- CORS enabled on the backend so the frontend can call it directly

## Troubleshooting

- **"backend offline" badge** → uvicorn isn't running, or it's on a different
  port than `API_BASE` in `script.js`.
- **"No models loaded" / 503 on /predict** → you haven't copied the `.keras`
  files into `backend/models/` yet.
- **Comparison charts empty** → copy `model_comparison.csv` into `backend/results/`.
- **Predictions look off / always same class** → make sure `clean_text()` in
  `backend/main.py` matches the notebook's preprocessing exactly (it's copied
  verbatim already, but double check if you edited the notebook later).
