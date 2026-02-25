from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import joblib
import shutil

from utils.config import MODEL_PATH, PREPROCESSOR_PATH, SAMPLE_NEW_DATA_PATH
from data.data_preprocessor import inference_transform

# -------------------------------------------------
# Globals 
# -------------------------------------------------
model = None
preprocessor = None


class BatchPredictionResponse(BaseModel):
    n_rows: int
    predictions: list[float]


# -------------------------------------------------
# Lifespan: load artifacts once
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor not found at {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("✅ Model & preprocessor loaded")
    yield
    print("🛑 API shutting down")


# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/health")
def health_check():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "ok",
        "model_loaded": True,
        "preprocessor_loaded": True,
    }


# -------------------------------------------------
# Batch Prediction/Inference 
# -------------------------------------------------
@app.post("/predict-csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)):

    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        X_processed = inference_transform(df, preprocessor)
        predictions = model.predict(X_processed)

        return BatchPredictionResponse(
            n_rows=len(predictions),
            predictions=predictions.tolist()
        )

    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing column: {e}")
    except Exception:
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# -------------------------------------------------
# Upload Labeled Data (for Drift Monitoring and Retraining)
# -------------------------------------------------
@app.post("/upload-labeled-data")
async def upload_labeled_data(file: UploadFile = File(...)):
    """
    Save labeled CSV to SAMPLE_NEW_DATA_PATH.
    Monitor service will detect drift & retrain if needed.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        SAMPLE_NEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        with SAMPLE_NEW_DATA_PATH.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "saved_to": str(SAMPLE_NEW_DATA_PATH),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


if __name__ == "__main__":
    print("HELLO WORLD")
