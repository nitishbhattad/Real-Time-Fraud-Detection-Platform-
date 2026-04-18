# ~/Desktop/kafka-fraud-detection/api_server.py
# ============================================================
# 🛡️ Fraud Detection API — FastAPI Microservice
# ============================================================
# Run:  uvicorn api_server:app --reload --port 8000
# Docs: http://localhost:8000/docs  (Swagger UI)
# ============================================================

import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BASE_DIR = "/Users/nitishbhattad/Desktop/kafka-fraud-detection"
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_paysim_rf.pkl")

FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
THRESHOLD = 0.9

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print(f"🔄 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")

# ---------------------------------------------------
# APP
# ---------------------------------------------------

app = FastAPI(
    title="🛡️ Fraud Detection API",
    description="""
    Real-time fraud scoring microservice for the Fraud Detection Platform.
    
    **Features:**
    - Score individual transactions via REST API
    - Batch scoring for multiple transactions
    - SHAP-based feature explanations
    - Health check endpoint
    
    **Architecture:** Kafka → Spark → **This API** → Dashboard
    
    Built by Nitish Bhattad & Sowmika Dinesh | DSC 550
    """,
    version="1.0.0",
)

# Allow CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------

class Transaction(BaseModel):
    """A single financial transaction to be scored."""
    transaction_id: Optional[str] = Field(None, example="txn_001")
    type: str = Field(..., example="CASH_OUT", description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
    amount: float = Field(..., example=181000.0, description="Transaction amount in dollars")
    oldbalanceOrg: float = Field(..., example=181000.0, description="Sender's balance before transaction")
    newbalanceOrig: float = Field(..., example=0.0, description="Sender's balance after transaction")
    oldbalanceDest: float = Field(..., example=0.0, description="Receiver's balance before transaction")
    newbalanceDest: float = Field(..., example=181000.0, description="Receiver's balance after transaction")


class FraudScore(BaseModel):
    """Fraud scoring result for a transaction."""
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    risk_level: str
    inference_time_ms: float
    top_features: dict


class BatchRequest(BaseModel):
    """Batch of transactions to score."""
    transactions: list[Transaction]


class BatchResponse(BaseModel):
    """Batch scoring results."""
    results: list[FraudScore]
    total_scored: int
    total_flagged: int
    batch_time_ms: float


class HealthResponse(BaseModel):
    """API health status."""
    status: str
    model_loaded: bool
    model_type: str
    threshold: float
    features: list[str]


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def classify_risk(probability: float) -> str:
    """Map fraud probability to human-readable risk level."""
    if probability >= 0.9:
        return "CRITICAL"
    elif probability >= 0.7:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "MINIMAL"


def get_feature_contributions(transaction_df: pd.DataFrame) -> dict:
    """
    Get feature importance for this specific prediction
    using the model's built-in feature importances as a proxy.
    """
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    # Transform the input
    X_transformed = preprocessor.transform(transaction_df)

    # Get feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(["type"]).tolist()
    num_names = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    all_names = cat_names + num_names

    # Feature importances * feature values = approximate contribution
    importances = clf.feature_importances_

    if hasattr(X_transformed, 'toarray'):
        X_array = X_transformed.toarray()[0]
    else:
        X_array = np.array(X_transformed)[0]

    contributions = {}
    for name, imp, val in zip(all_names, importances, X_array):
        contributions[name] = round(float(imp * abs(val)), 4)

    # Sort by absolute contribution and return top 6
    sorted_contribs = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
    )
    return sorted_contribs


def score_single(txn: Transaction) -> FraudScore:
    """Score a single transaction."""
    start = time.time()

    # Build DataFrame
    df = pd.DataFrame([{
        "type": txn.type,
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
    }])

    # Predict
    probability = float(model.predict_proba(df)[:, 1][0])
    is_fraud = probability >= THRESHOLD

    # Feature contributions
    top_features = get_feature_contributions(df)

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return FraudScore(
        transaction_id=txn.transaction_id,
        fraud_probability=round(probability, 4),
        is_fraud=is_fraud,
        threshold_used=THRESHOLD,
        risk_level=classify_risk(probability),
        inference_time_ms=elapsed_ms,
        top_features=top_features,
    )


# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """API root — redirect to docs."""
    return {
        "message": "🛡️ Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
        "score_endpoint": "POST /score",
        "batch_endpoint": "POST /score/batch",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """Check if the API and model are healthy."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type=type(model.named_steps["clf"]).__name__,
        threshold=THRESHOLD,
        features=FEATURES,
    )


@app.post("/score", response_model=FraudScore, tags=["Scoring"])
def score_transaction(txn: Transaction):
    """
    Score a single transaction for fraud.

    Returns fraud probability, binary decision, risk level,
    inference time, and top contributing features.
    """
    try:
        return score_single(txn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
def score_batch(batch: BatchRequest):
    """
    Score multiple transactions in a single request.

    Returns individual results plus batch statistics.
    """
    start = time.time()

    try:
        results = [score_single(txn) for txn in batch.transactions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return BatchResponse(
        results=results,
        total_scored=len(results),
        total_flagged=sum(1 for r in results if r.is_fraud),
        batch_time_ms=elapsed_ms,
    )


@app.post("/explain", tags=["Explainability"])
def explain_transaction(txn: Transaction):
    """
    Get a detailed explanation of why a transaction was flagged.

    Uses SHAP values for feature-level explanations.
    """
    try:
        import shap

        df = pd.DataFrame([{
            "type": txn.type,
            "amount": txn.amount,
            "oldbalanceOrg": txn.oldbalanceOrg,
            "newbalanceOrig": txn.newbalanceOrig,
            "oldbalanceDest": txn.oldbalanceDest,
            "newbalanceDest": txn.newbalanceDest,
        }])

        probability = float(model.predict_proba(df)[:, 1][0])

        # SHAP explanation
        def predict_fn(X_array):
            X_df = pd.DataFrame(X_array, columns=FEATURES)
            return model.predict_proba(X_df)[:, 1]

        # Use the transaction itself as background (fast approximation)
        explainer = shap.KernelExplainer(predict_fn, df.values)
        shap_values = explainer.shap_values(df.values)[0]

        contributions = {feat: round(float(sv), 4) for feat, sv in zip(FEATURES, shap_values)}
        sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

        return {
            "transaction_id": txn.transaction_id,
            "fraud_probability": round(probability, 4),
            "is_fraud": probability >= THRESHOLD,
            "risk_level": classify_risk(probability),
            "shap_contributions": sorted_contribs,
            "explanation": [
                f"{feat} {'increases' if val > 0 else 'decreases'} fraud risk by {abs(val):.4f}"
                for feat, val in sorted_contribs.items()
            ],
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="SHAP not installed. Run: pip install shap")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
