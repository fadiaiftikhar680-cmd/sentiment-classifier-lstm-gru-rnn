# 🎬 CineSense — Movie Review Sentiment Analyzer

**CineSense** is a deep learning-powered web application that analyzes movie reviews and predicts whether the review is **Positive** or **Negative**, along with a confidence score.

The project compares three recurrent neural network architectures — **Simple RNN, LSTM, and GRU** — trained from scratch on the **IMDb 50K Movie Reviews dataset**. The models are served through a **FastAPI backend** and connected to a custom **HTML/CSS/JavaScript frontend** with a cinema-inspired user interface.

> ⭐ **GRU is used as the default production model** because it provides an excellent balance between accuracy, recall, and training efficiency.

---

## ✨ Features

* 🎥 Real-time movie review sentiment prediction
* 🧠 Three deep learning architectures:

  * Simple RNN
  * LSTM
  * GRU
* 🔄 Switch between models directly from the frontend
* 📊 Model comparison dashboard
* 📈 Accuracy, Precision, Recall, F1-Score and ROC-AUC metrics
* ⏱️ Training and inference time comparison
* 🧹 Complete NLP preprocessing pipeline
* 🔤 Keras tokenization and sequence padding
* ⚡ FastAPI REST API
* 📚 Interactive API documentation with Swagger
* 💻 Custom cinema-ticket themed frontend
* 🌐 CORS-enabled backend
* ❤️ Health and model metadata endpoints
* 📦 Native `.keras` model format
* 🔬 Full training and experimentation notebook included

---

## 🧠 How CineSense Works

```text
                    ┌──────────────────────┐
                    │   Movie Review Text  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   NLP Preprocessing  │
                    │                      │
                    │ • Lowercasing        │
                    │ • HTML Removal       │
                    │ • URL Removal        │
                    │ • Punctuation Removal│
                    │ • Digit Removal      │
                    │ • Stopword Removal   │
                    │ • Lemmatization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Tokenization    │
                    │   Vocabulary: 20,000 │
                    │   Max Length: 200    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           ┌────────┐     ┌────────┐     ┌────────┐
           │  RNN   │     │  LSTM  │     │  GRU   │
           └────┬───┘     └────┬───┘     └────┬───┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Sentiment Prediction │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Positive / Negative  │
                    │ + Confidence Score    │
                    └──────────────────────┘
```

---

## 📊 Dataset

CineSense uses the **IMDb 50K Movie Reviews dataset**, containing binary sentiment labels:

* **Positive**
* **Negative**

### Dataset Split

| Split      |    Samples |
| ---------- | ---------: |
| Training   |     34,707 |
| Validation |      7,437 |
| Testing    |      7,438 |
| **Total**  | **49,582** |

### NLP Preprocessing

Each review follows the same preprocessing pipeline during training and prediction:

1. Convert text to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove punctuation
5. Remove digits
6. Remove stopwords
7. Lemmatize words
8. Tokenize using Keras Tokenizer
9. Pad/truncate sequences

> ⚠️ The preprocessing pipeline must remain consistent between training and prediction to maintain model performance.

---

## 🧠 Model Architecture

Three recurrent neural network architectures were trained using the same dataset splits and preprocessing pipeline.

### 1. Simple RNN

A basic recurrent neural network used as a baseline model.

### 2. LSTM

Long Short-Term Memory network designed to capture longer-term dependencies in sequential text.

### 3. GRU

Gated Recurrent Unit network that provides a simpler and computationally efficient alternative to LSTM while maintaining strong performance.

### Common Configuration

| Parameter               | Value              |
| ----------------------- | ------------------ |
| Vocabulary Size         | 20,000             |
| Maximum Sequence Length | 200                |
| Embedding Dimension     | 128                |
| Recurrent Units         | 64                 |
| Architectures           | RNN, LSTM, GRU     |
| Framework               | TensorFlow / Keras |

---

# 📈 Model Comparison

The three models were trained and evaluated under the same experimental setup.

| Model      |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC | Training Time |
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: | ------------: |
| 🥇 **GRU** | **87.79%** |     83.01% | **95.15%** | **88.67%** | **94.90%** |        530.1s |
| LSTM       | **87.97%** | **88.06%** |     87.95% |     88.00% |     94.38% |        785.9s |
| RNN        |     50.38% |     54.32% |      7.07% |     12.51% |     51.27% |    **108.9s** |

### 🏆 Why GRU?

Although LSTM achieved a slightly higher accuracy, **GRU was selected as the default production model** because it:

* Achieves almost the same accuracy as LSTM
* Achieves significantly higher recall
* Achieves the highest F1-score
* Achieves the highest ROC-AUC
* Requires less training time than LSTM
* Provides a strong balance between performance and efficiency

The **Simple RNN** performed poorly on this dataset, demonstrating the difficulty of basic RNNs in learning long-range dependencies in lengthy movie reviews.

---

# 🔬 Deep Learning Training

All three models were trained from scratch using **TensorFlow/Keras**.

The complete experimentation process is available in:

```text
Sentiment_Analysis_RNN_LSTM_GRU.ipynb
```

The notebook contains:

* Dataset loading
* Exploratory analysis
* Text preprocessing
* Tokenization
* Sequence padding
* Model creation
* Model training
* Validation
* Performance evaluation
* Model comparison
* Metric visualization
* Model saving

---

# 🏗️ Project Architecture

```text
CineSense
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── best_model.keras
│   │   ├── gru.keras
│   │   ├── lstm.keras
│   │   └── rnn.keras
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.pickle
│   │   ├── label_encoder.pickle
│   │   └── config.json
│   │
│   └── results/
│       └── model_comparison.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── Sentiment_Analysis_RNN_LSTM_GRU.ipynb
```

---

# ⚙️ Tech Stack

### Deep Learning

* TensorFlow
* Keras
* RNN
* LSTM
* GRU

### NLP

* NLTK
* Stopword Removal
* Tokenization
* WordNet Lemmatization

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Data Processing

* NumPy
* Pandas

### Frontend

* HTML5
* CSS3
* Vanilla JavaScript

### Model Format

* Keras `.keras`

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python 3.11+
* pip
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/fadiaiftikhar680-cmd/sentiment-classifier-lstm-gru-rnn.git

cd sentiment-classifier-lstm-gru-rnn
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
cd backend

pip install -r requirements.txt
```

---

# ⚡ Run the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 💻 Run the Frontend

Open another terminal and navigate to the frontend:

```bash
cd frontend
```

Start a local server:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

You can also open `index.html` directly in a browser.

---

# 🔌 API Endpoints

| Method | Endpoint            | Description                                  |
| ------ | ------------------- | -------------------------------------------- |
| GET    | `/`                 | API status and loaded models                 |
| GET    | `/health`           | Health check                                 |
| GET    | `/model-info`       | Training configuration and model information |
| GET    | `/model-comparison` | RNN vs LSTM vs GRU metrics                   |
| POST   | `/predict`          | Predict sentiment for a review               |

---

# 🧪 Prediction API

### Request

```http
POST /predict
```

```json
{
  "text": "This movie was absolutely brilliant, the acting was superb!",
  "model": "GRU"
}
```

### Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9732,
  "probability_positive": 0.9732,
  "probability_negative": 0.0268,
  "model_used": "GRU",
  "inference_time_ms": 12.4,
  "cleaned_text": "movie absolutely brilliant acting superb"
}
```

---

# 🎬 Example Reviews

### Positive

```text
This movie was absolutely brilliant. The acting was superb and the story was incredibly engaging.
```

**Prediction:** Positive

### Negative

```text
The movie was extremely boring. The story was weak and the acting was disappointing.
```

**Prediction:** Negative

### Another Positive Example

```text
I loved every minute of this film. The characters were well written and the ending was fantastic.
```

**Prediction:** Positive

### Another Negative Example

```text
This was a complete waste of time. The plot was confusing and the performances were poor.
```

**Prediction:** Negative

---

# 🔐 Model & Preprocessing Consistency

CineSense uses the **same preprocessing pipeline during training and inference**.

This includes:

* HTML removal
* URL removal
* Punctuation removal
* Digit removal
* Stopword removal
* Lemmatization
* Tokenization
* Sequence padding

> ⚠️ Changing the preprocessing logic in `main.py` without updating the training pipeline can negatively affect prediction quality.

---

# 📦 Model Files

The project contains three trained models:

```text
gru.keras
lstm.keras
rnn.keras
```

The default production model is:

```text
best_model.keras
```

`best_model.keras` is a copy of the GRU model and is loaded automatically when no other model is explicitly selected.

Model files are managed using **Git LFS** because of their size.

---

# 📚 API Documentation

FastAPI automatically generates interactive API documentation.

After starting the backend, visit:

```text
http://localhost:8000/docs
```

You can test endpoints directly from the Swagger interface.

---

# 🔄 Prediction Flow

```text
User enters movie review
          ↓
Frontend sends POST request
          ↓
FastAPI receives review
          ↓
Text preprocessing
          ↓
Tokenization
          ↓
Sequence padding
          ↓
Selected RNN/LSTM/GRU model
          ↓
Probability calculation
          ↓
Positive / Negative classification
          ↓
Confidence score
          ↓
Frontend displays result
```

---

# 🎯 Project Goals

CineSense was developed to demonstrate how **deep learning and Natural Language Processing can be combined to build a complete end-to-end sentiment analysis system**.

The project focuses on:

* Understanding recurrent neural networks
* Comparing RNN, LSTM, and GRU
* Applying NLP preprocessing techniques
* Training deep learning models from scratch
* Building REST APIs with FastAPI
* Connecting machine learning models to a frontend
* Evaluating models using multiple performance metrics
* Deploying a complete ML application architecture

---

# 📌 Key Takeaways

### 🥇 Best Production Model

**GRU**

### 📊 Best Accuracy

**LSTM — 87.97%**

### 🎯 Best Recall

**GRU — 95.15%**

### ⭐ Best F1 Score

**GRU — 88.67%**

### 📈 Best ROC-AUC

**GRU — 94.90%**

### ⚡ Fastest Training

**Simple RNN — 108.9 seconds**

The results demonstrate that **GRU provides the best overall balance for this application**, while the comparison with LSTM and Simple RNN highlights the practical differences between recurrent architectures.

---

# 👩‍💻 Author

**Fadia Iftikhar**

BS Artificial Intelligence Student

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
