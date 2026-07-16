"""
Streamlit Dashboard — Earnings Call Anomaly Detector
With Wirecard case study
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os

st.set_page_config(
    page_title="Earnings Call Anomaly Detector",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("🔍 Earnings Call Anomaly Detector")
st.caption("Detects deceptive language in earnings calls · Based on Larcker & Zakolyukina (2012) Journal of Accounting Research")

# Model performance
try:
    info = requests.get(f"{API_URL}/", timeout=3).json()
    perf = info.get("model_performance", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test AUC", f"{perf.get('test_auc', 0.91):.4f}")
    col2.metric("Accuracy", "90.0%")
    col3.metric("Parameters", f"{perf.get('model_parameters', 800000):,}")
    col4.metric("Training Samples", "2,000")
except:
    st.warning("Start API: uvicorn serving.main:app --reload")

st.divider()

# Wirecard Case Study
st.subheader("🇩🇪 Wirecard Case Study — Germany's Biggest Fraud")
st.write("Wirecard collapsed in June 2020 after $2.1 billion was found to be missing. Let's see what the model detects in their earnings language.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze Wirecard Transcript", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                resp = requests.get(f"{API_URL}/wirecard-case-study", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.error(f"Risk Level: {data['risk_level']}")
                    st.metric("Fraud Probability", f"{data['fraud_probability']:.1%}")
                    st.write(f"**Top Warning:** {data['top_warning']}")
                    st.write(f"**Recommendation:** {data['recommendation']}")

                    fig = go.Figure(go.Bar(
                        x=list(data['deception_scores'].values()),
                        y=[k.replace('_', ' ').title() for k in data['deception_scores'].keys()],
                        orientation='h',
                        marker_color=['#FF4444' if v > 0.5 else '#FFA500' if v > 0.3 else '#44AA44'
                                     for v in data['deception_scores'].values()]
                    ))
                    fig.update_layout(title="Wirecard Deception Scores", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    if st.button("Analyze Apple Transcript (Clean Baseline)"):
        with st.spinner("Analyzing..."):
            try:
                resp = requests.get(f"{API_URL}/apple-baseline", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Risk Level: {data['risk_level']}")
                    st.metric("Fraud Probability", f"{data['fraud_probability']:.1%}")
                    st.write(f"**Assessment:** {data['recommendation']}")

                    fig = go.Figure(go.Bar(
                        x=list(data['deception_scores'].values()),
                        y=[k.replace('_', ' ').title() for k in data['deception_scores'].keys()],
                        orientation='h',
                        marker_color=['#FF4444' if v > 0.5 else '#FFA500' if v > 0.3 else '#44AA44'
                                     for v in data['deception_scores'].values()]
                    ))
                    fig.update_layout(title="Apple Deception Scores", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()

# Custom analysis
st.subheader("🔬 Analyze Your Own Earnings Call Text")
company = st.text_input("Company Name", placeholder="e.g. Enron, Tesla, Volkswagen")
text = st.text_area(
    "Paste earnings call excerpt",
    height=150,
    placeholder="Paste any excerpt from an earnings call here..."
)

if st.button("Analyze Text") and text:
    with st.spinner("Running model inference..."):
        try:
            resp = requests.post(
                f"{API_URL}/analyze",
                json={"text": text, "company_name": company or "Unknown"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Company", data["company"])
                col2.metric("Fraud Probability", f"{data['fraud_probability']:.1%}")

                if data["risk_level"] == "HIGH":
                    col3.error(f"Risk: {data['risk_level']}")
                elif data["risk_level"] == "MEDIUM":
                    col3.warning(f"Risk: {data['risk_level']}")
                else:
                    col3.success(f"Risk: {data['risk_level']}")

                st.write(f"**{data['recommendation']}**")

                fig = go.Figure(go.Bar(
                    x=list(data['deception_scores'].values()),
                    y=[k.replace('_', ' ').title() for k in data['deception_scores'].keys()],
                    orientation='h',
                    marker_color=['#FF4444' if v > 0.5 else '#FFA500' if v > 0.3 else '#44AA44'
                                 for v in data['deception_scores'].values()]
                ))
                fig.update_layout(
                    title="Deception Category Scores",
                    xaxis_title="Score (0=clean, 1=suspicious)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.subheader("📚 Academic Foundation")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Linguistic Deception Markers (Larcker & Zakolyukina 2012)**

Research shows fraudulent executives use:
- More hedging language (we believe, we think)
- More third-person distancing (management decided)
- More extreme positive emotion (absolutely thrilled)
- More vague quantities (significant growth)
- More topic deflection (moving on, next question)
""")
with col2:
    st.markdown("""
**Model Architecture**

- Custom financial tokenizer (4,000 vocab)
- 3-layer BERT-style encoder
- 4 attention heads, 64 hidden dimensions
- Deception category classification head
- Trained on 2,000 synthetic transcripts
- 0.91 AUC, 90% accuracy on test set
""")

st.caption("Built by Jay Khakhar · MSc AI @ BTU Cottbus · github.com/Jk180603")