import os
import sys
from typing import Dict, Tuple
from pathlib import Path

import torch

# Ensure project root (which contains the `model` package) is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model_architecture import HybridBertCnnLstmAttention
from model.bert_embeddings import BertEmbeddingGenerator
from model.preprocess import clean_text
from explainability import extract_suspicious_patterns


MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fake_recruitment_detector.pth")


class PredictionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bertphish_path = Path(PROJECT_ROOT) / "model" / "bertphish-transformers-default-v1"
        self.bert_model_name = str(bertphish_path) if bertphish_path.exists() else "bert-base-uncased"

        self.bert_gen = BertEmbeddingGenerator(model_name=self.bert_model_name)
        self.model = HybridBertCnnLstmAttention(bert_model_name=self.bert_model_name)

        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, email_text: str) -> Tuple[float, float]:
        cleaned = clean_text(email_text)
        encodings = self.bert_gen.encode_batch([cleaned])
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs["probs"].cpu().numpy()[0]

        # label 0 = Genuine, label 1 = Fake
        genuine_prob = float(probs[0])
        fraud_prob = float(probs[1])
        return fraud_prob, genuine_prob


_ENGINE = PredictionEngine()


def risk_level_from_probability(fraud_probability: float) -> str:
    pct = fraud_probability * 100
    if pct < 40:
        return "Low"
    elif pct < 70:
        return "Medium"
    return "High"


def analyze_email(email_text: str, subject: str = "N/A", sender_email: str = "") -> Dict:
    fraud_prob, genuine_prob = _ENGINE.predict(email_text)

    explain = extract_suspicious_patterns(email_text=email_text, sender_email=sender_email)

    adjusted_fraud_prob = min(fraud_prob + explain.get("risk_score_bonus", 0.0), 0.999)
    adjusted_genuine_prob = max(1.0 - adjusted_fraud_prob, 0.001)

    # High-risk alert override (PPT: alert notification for high-risk emails)
    if explain.get("high_risk"):
        adjusted_fraud_prob = max(adjusted_fraud_prob, 0.9)
        adjusted_genuine_prob = min(adjusted_genuine_prob, 0.1)

    risk = risk_level_from_probability(adjusted_fraud_prob)
    prediction_label = "Fake" if adjusted_fraud_prob >= 0.5 else "Genuine"

    alerts = []
    if risk == "High":
        alerts.append("High-risk recruitment fraud indicators detected. Do not send money or personal documents.")

    return {
        "subject": subject,
        "prediction": prediction_label,
        "risk_level": risk,
        "fraud_probability": round(adjusted_fraud_prob * 100, 2),
        "genuine_probability": round(adjusted_genuine_prob * 100, 2),
        "reasons": explain.get("reasons", []),
        "suspicious_keywords": explain.get("suspicious_keywords", []),
        "alert": risk == "High",
        "alerts": alerts,
    }
