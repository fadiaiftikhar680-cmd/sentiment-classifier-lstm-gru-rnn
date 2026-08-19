CineSense — Movie Review Sentiment Analyzer
Deep learning web app that predicts whether a movie review is positive or negative, using three different recurrent neural network architectures (RNN, LSTM, GRU) trained from scratch and served through a FastAPI backend with a custom HTML/CSS/JS frontend.

Overview
CineSense takes a raw movie review as text input, cleans and tokenizes it, and runs it through a trained deep learning model to classify the sentiment as Positive or Negative, along with a confidence score. The project compares three RNN-family architectures on the same dataset so their performance can be evaluated side by side, and lets the best-performing model (GRU) serve predictions by default — while still allowing any of the three to be selected manually from the UI.

Features
Real-time sentiment prediction from free-text movie reviews via a REST API
Three trained models (Simple RNN, LSTM, GRU) — switch between them from the frontend
Model comparison dashboard — Accuracy, Precision, Recall, F1-score, ROC-AUC, and training/inference time for every model
NLP preprocessing pipeline — HTML/URL stripping, punctuation & digit removal, stopword removal, and lemmatization identical to the training pipeline (so predictions match training-time behavior exactly)
FastAPI backend with automatic interactive API docs (/docs)
Custom-designed frontend ("cinema ticket" themed UI) — no frameworks, pure HTML/CSS/JS
CORS-enabled API so the frontend can be hosted separately from the backend
Health & metadata endpoints to check which models are loaded and inspect training configuration
Tech Stack
Layer	Technology
Deep Learning	TensorFlow / Keras
NLP Preprocessing	NLTK (stopwords, tokenizer, WordNet lemmatizer)
Backend API	FastAPI + Uvicorn
Data Handling	NumPy, Pandas
Validation	Pydantic
Frontend	HTML5, CSS3, Vanilla JavaScript
Model Format	.keras (native Keras format)
Dataset & Training
Dataset: IMDb 50K Movie Reviews (binary sentiment: positive / negative)
Total samples: 49,582
Training: 34,707
Validation: 7,437
Testing: 7,438
Preprocessing: lowercasing, HTML tag removal, URL removal, punctuation/digit removal, stopword removal, lemmatization
Tokenization: Keras Tokenizer with a vocabulary size of 20,000, sequences padded/truncated to a max length of 200
Embedding dimension: 128
Recurrent units: 64
Architectures trained: Simple RNN, LSTM, GRU — all trained on identical splits and preprocessing for a fair comparison
Full training code, experimentation, and evaluation live in the notebook: Sentiment_Analysis_RNN_LSTM_GRU.ipynb

Model Comparison Results
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC	Training Time (s)
GRU ⭐ (best)	87.79%	83.01%	95.15%	88.67%	94.90%	530.1
LSTM	87.97%	88.06%	87.95%	88.00%	94.38%	785.9
RNN	50.38%	54.32%	7.07%	12.51%	51.27%	108.9
Key takeaway: GRU is used as the default production model — it matches LSTM's accuracy while training ~1.5x faster and achieving a notably higher recall. The plain RNN struggles with long-range dependencies in review text and performs close to random guessing, illustrating the classic vanishing-gradient limitation of simple RNNs on long sequences.

Project Structure
sentiment-app/
├── backend/
│   ├── main.py                  # FastAPI app — loads models, exposes /predict
│   ├── requirements.txt
│   ├── models/
│   │   ├── best_model.keras     # GRU (default/production model)
│   │   ├── gru.keras
│   │   ├── lstm.keras
│   │   └── rnn.keras
│   ├── tokenizer/
│   │   ├── tokenizer.pickle
│   │   ├── label_encoder.pickle
│   │   └── config.json          # vocab size, max_len, model paths, label mapping
│   └── results/
│       └── model_comparison.csv # metrics used for the /model-comparison endpoint
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── Sentiment_Analysis_RNN_LSTM_GRU.ipynb   # training & experimentation notebook
API Endpoints
Method	Endpoint	Description
GET	/	API status and currently loaded models
GET	/health	Health check
GET	/model-info	Training configuration and available models
GET	/model-comparison	Returns metrics table for RNN vs LSTM vs GRU
POST	/predict	Predicts sentiment for a given review text
Example request:

POST /predict
{
  "text": "This movie was absolutely brilliant, the acting was superb!",
  "model": "GRU"
}
Example response:

{
  "sentiment": "positive",
  "confidence": 0.9732,
  "probability_positive": 0.9732,
  "probability_negative": 0.0268,
  "model_used": "GRU",
  "inference_time_ms": 12.4,
  "cleaned_text": "movie absolutely brilliant acting superb"
}
Getting Started
Prerequisites
Python 3.11+
pip
1. Clone the repository
git clone https://github.com/fadiaiftikhar680-cmd/sentiment-classifier-lstm-gru-rnn.git
cd sentiment-classifier-lstm-gru-rnn
2. Set up the backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
3. Run the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
The API will be available at http://localhost:8000, with interactive docs at http://localhost:8000/docs.

4. Launch the frontend
Open frontend/index.html directly in a browser, or serve it with a simple local server:

cd frontend
python -m http.server 5500
Then visit http://localhost:5500.

Notes
Model files (.keras) are tracked with Git LFS due to their size (~30 MB each).
The text-cleaning function in main.py must stay identical to the one used in the training notebook — any mismatch will degrade prediction quality.
best_model.keras is a copy of the GRU model and is used automatically unless another model is explicitly requested in the API call.
Author
Fadia Iftikhar
