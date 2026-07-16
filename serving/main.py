"""
FastAPI serving for Earnings Call Anomaly Detector
"""
import torch
import json
import os
import sys
sys.path.insert(0, ".")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.detector import EarningsAnomalyDetector, FinancialTokenizer

app = FastAPI(
    title="Earnings Call Anomaly Detector",
    description="Detects deceptive language in earnings calls using transformer NLP",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model
tokenizer = FinancialTokenizer(vocab_size=4000, max_len=128)
tokenizer.load("model/tokenizer.json")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EarningsAnomalyDetector(
    vocab_size=4000, d_model=64, num_heads=4,
    num_layers=3, d_ff=256, max_len=128
).to(device)
model.load_state_dict(torch.load("model/checkpoints/best_model.pt", map_location=device))
model.eval()

DECEPTION_CATEGORIES = ["hedging", "excessive_certainty", "topic_avoidance", "distancing", "positive_framing"]

WIRECARD_SAMPLE = """
We are absolutely certain that our payment processing volumes
represent genuine customer transactions. Management has decided
to accelerate certain revenue recognition to better reflect
the underlying business momentum. We believe the business is
performing exceptionally well. Moving on, let's focus on
future opportunities rather than dwelling on past quarters.
The company is thrilled with these outstanding results.
"""

APPLE_SAMPLE = """
Revenue was $111.4 billion, up 21% year over year.
iPhone revenue was $65.6 billion. We set all-time records
in every geographic segment. Our gross margin was 39.8%.
We generated $38.8 billion in operating cash flow.
We returned $30 billion to shareholders. For Q2 guidance,
we expect revenue between $82.5 and $86.5 billion.
"""


class AnalyzeRequest(BaseModel):
    text: str
    company_name: str = "Unknown"


class AnalyzeResponse(BaseModel):
    company: str
    fraud_probability: float
    risk_level: str
    deception_scores: dict
    top_warning: str
    recommendation: str


@app.get("/")
def root():
    results = {}
    if os.path.exists("model/results.json"):
        with open("model/results.json") as f:
            results = json.load(f)
    return {
        "message": "Earnings Call Anomaly Detector",
        "model_performance": results,
        "academic_basis": "Larcker & Zakolyukina (2012), Journal of Accounting Research",
        "use_case": "Detect deceptive language patterns in earnings call transcripts",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short")

    ids = tokenizer.encode(req.text)
    input_tensor = torch.LongTensor([ids]).to(device)

    with torch.no_grad():
        out = model(input_tensor)

    fraud_prob = float(out["fraud_prob"].item())
    deception_scores = out["deception_scores"].squeeze().cpu().numpy().tolist()

    deception_dict = {
        cat: round(float(score), 3)
        for cat, score in zip(DECEPTION_CATEGORIES, deception_scores)
    }

    top_category = max(deception_dict, key=deception_dict.get)

    if fraud_prob > 0.7:
        risk_level = "HIGH"
        recommendation = "Flag for further investigation. Multiple deception markers detected."
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM"
        recommendation = "Monitor closely. Some linguistic anomalies present."
    else:
        risk_level = "LOW"
        recommendation = "Language patterns consistent with transparent communication."

    return AnalyzeResponse(
        company=req.company_name,
        fraud_probability=round(fraud_prob, 3),
        risk_level=risk_level,
        deception_scores=deception_dict,
        top_warning=f"High {top_category.replace('_', ' ')} detected",
        recommendation=recommendation,
    )


@app.get("/wirecard-case-study")
def wirecard():
    """Analyze Wirecard transcript as case study"""
    req = AnalyzeRequest(text=WIRECARD_SAMPLE, company_name="Wirecard AG")
    return analyze(req)


@app.get("/apple-baseline")
def apple():
    """Analyze Apple transcript as clean baseline"""
    req = AnalyzeRequest(text=APPLE_SAMPLE, company_name="Apple Inc")
    return analyze(req)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}