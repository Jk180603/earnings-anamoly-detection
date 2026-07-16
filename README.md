# Earnings Call Anomaly Detector

A BERT-style transformer trained **from scratch** in PyTorch to detect deceptive language patterns in earnings call transcripts. Built on academic research by Larcker & Zakolyukina (2012, Journal of Accounting Research).

**0.91 AUC · 90% Accuracy · 89% F1 on held-out test set.**

---

## What it does

Fraudulent executives use measurable linguistic patterns before financial scandals emerge. This model detects:

- **Hedging language** — "we believe", "it appears", "approximately"
- **Topic avoidance** — sudden subject changes, deflection phrases
- **Excessive certainty** — "absolutely guaranteed", "without doubt"
- **Third-person distancing** — "management decided", "it was determined"
- **Positive framing spikes** — unusually high praise language

Validated on a Wirecard case study — the model flags language anomalies in their earnings calls that preceded the 2020 collapse.

---

## Results

| Metric | Value |
|---|---|
| Test AUC | 0.9065 |
| Accuracy | 90% |
| F1 Score | 89% |
| Precision (fraud) | 90% |
| Recall (fraud) | 85% |
| Training samples | 2,000 |
| Model parameters | ~800K |

---

## Architecture

Custom BERT-style encoder built from scratch in PyTorch:

```
Input: token ids (batch, seq_len=128)

↓ Token Embedding (4,000 vocab → 64 dims)
↓ Positional Embedding
↓ Transformer Block × 3
    ├── Multi-Head Self-Attention (4 heads)
    ├── LayerNorm + Residual
    ├── FeedForward (64 → 256 → 64, GELU)
    └── LayerNorm + Residual
↓ CLS Token Extraction
↓ Fraud Classification Head (64 → 32 → 1)
↓ Deception Category Head (64 → 5 categories)

Output: fraud probability + 5 deception category scores
```

**Custom financial tokenizer** trained on financial corpus vocabulary (4,000 tokens).

---

## Deception Categories

The model scores 5 linguistic deception dimensions per text:

1. **Hedging** — uncertainty language
2. **Excessive Certainty** — overconfident language
3. **Topic Avoidance** — deflection patterns
4. **Distancing** — third-person language
5. **Positive Framing** — unusual praise spikes

---

## Getting Started

```bash
git clone https://github.com/Jk180603/earnings-anomaly-detector
cd earnings-anomaly-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Download data and train:**
```bash
python data/earnings_download.py
python data/get_proper_dataset.py
python model/train.py
```

**Start API:**
```bash
uvicorn serving.main:app --reload
```

**Start dashboard:**
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` — click "Analyze Wirecard Transcript" to see the model flag deceptive language patterns.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Model info and performance |
| `/analyze` | POST | Analyze any earnings call text |
| `/wirecard-case-study` | GET | Wirecard fraud analysis |
| `/apple-baseline` | GET | Clean company baseline |
| `/health` | GET | Health check |

---

## Project Structure

```
earnings-anomaly-detector/
├── data/
│   ├── earnings_download.py     # Data collection
│   └── get_proper_dataset.py    # Synthetic dataset with noise
├── model/
│   ├── detector.py              # BERT-style encoder from scratch
│   ├── train.py                 # Training loop with MLflow
│   └── checkpoints/             # Saved model weights
├── serving/
│   └── main.py                  # FastAPI with Wirecard case study
├── dashboard/
│   └── app.py                   # Streamlit visualization
└── requirements.txt
```

---

## Academic Foundation

Based on: **Larcker, D.F. & Zakolyukina, A.A. (2012)**
*"Detecting Deceptive Discussions in Conference Calls"*
Journal of Accounting Research, 50(2), 495-540.

---

Built by [Jay Khakhar](https://github.com/Jk180603) · MSc AI @ BTU Cottbus
