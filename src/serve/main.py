from fastapi import FastAPI, HTTPException
from typing import Dict
import uvicorn
from contextlib import asynccontextmanager
from collections import deque
import pandas as pd

from src.logger import logger
from src.monitoring.drift_detector import (
    detect_univariate_drift,
    detect_multivariate_drift, 
    detect_concept_drift,
)
from .predict import get_model_info, predict_churn
from .schema import PredictionRequest
from fastapi.middleware.cors import CORSMiddleware
# Set MLflow tracking URI for model loading
import mlflow
import os


# Rolling window of recent inference payloads 
INFERENCE_BUFFER_SIZE = 1000
_inference_buffer: deque = deque(maxlen=INFERENCE_BUFFER_SIZE)
_labels_buffer: deque = deque(maxlen=INFERENCE_BUFFER_SIZE)
_model_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_ready
    logger.info("Starting Prediction API server.......")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
    mlflow.set_tracking_uri(tracking_uri)
    get_model_info(file_path='models/all_models_metric.json')
    _model_ready = True
    yield
    logger.info("Shutting down churn prediction service...")  
   

app = FastAPI(
    title="Churn Prediction API",
    description="Churn prediction service with drift monitoring",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get('/', tags=['Health'])
async def root() -> Dict[str, str]:
    return {"message": "Churn prediction API is Running..."}


@app.post('/predict', tags=['Prediction'])
async def predict(request: PredictionRequest):
    try:
        payload = request.model_dump()
        result = predict_churn(input=__import__("json").dumps(payload))

        # Record the request for downstream drift monitoring
        _inference_buffer.append(payload)
        logger.info(f"Buffered inference sample (buffer size: {len(_inference_buffer)})")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal error")


@app.get('/drift', tags=['Monitoring'])
async def drift_status() -> Dict:
    """Report drift of live traffic vs. the training reference profile."""
    if len(_inference_buffer) < 50:
        return {
            "status": "insufficient_data",
            "samples": len(_inference_buffer),
            "minimum_required": 50,
        }

    current = pd.DataFrame(list(_inference_buffer))
    reference = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

    reports = detect_univariate_drift(reference, current)
    multivariate = detect_multivariate_drift(
        reference, current,
        features=["MonthlyCharges", "tenure", "Contract", "PaymentMethod", "InternetService"],
    )
    drifted_features = [r.feature for r in reports if r.drifted]

    if drifted_features or multivariate.get("drifted"):
        logger.warning(
            f"Drift detected | univariate={drifted_features} | "
            f"multivariate_auc={multivariate.get('auc')}"
        )
    else:
        logger.info("No drift detected in current window")

    return {
        "status": "drift_detected" if (drifted_features or multivariate.get("drifted")) else "stable",
        "samples": len(current),
        "drifted_features": drifted_features,
        "univariate": [r.to_dict() for r in reports],
        "multivariate": multivariate,
        "retrain_recommended": bool(drifted_features or multivariate.get("drifted")),
    }


@app.post('/drift/label', tags=['Monitoring'])
async def record_label(request: Dict):
    """Feedback endpoint: ground truth arrives later, enabling concept-drift checks."""
    _labels_buffer.append(request)
    return {"status": "recorded", "buffered": len(_labels_buffer)}


if __name__ == "__main__":
    # Run the application with uvicorn.
    #
    # NOTE: `reload=True` makes uvicorn watch the working directory with watchfiles.
    # Because this project writes runtime artifacts INSIDE the project root
    # (logs/, models/, mlflow.db, .venv/ ...), every single log write was
    # detected as a "change" and triggered a server restart -> infinite loop
    # of "1 change detected" messages.
    #
    # Fix: watch ONLY the source code folders and only *.py files.
    uvicorn.run(
        "src.serve.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src", "main.py"],
        reload_includes=["*.py"],
        reload_excludes=[
            "logs/*",
            "models/*",
            "data/*",
            "notebooks/*",
            "reports/*",
            ".venv/*",
            "*.pyc",
            "__pycache__/*",
            "*.log",
            "mlflow.db",
        ],
        log_level="info"
    )
