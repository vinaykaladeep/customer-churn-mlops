# src/app.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np

from typing import List

from src.serving.model_loader import load_production_model
from src.serving.preprocessor_loader import load_preprocessor
from src.serving.inference_service import InferenceService

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, threshold = load_production_model()
        app.state.model = model
        app.state.threshold = threshold
        yield
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

# app = FastAPI(lifespan=lifespan)

app = FastAPI()
model, threshold = load_production_model()
preprocessor = load_preprocessor()
inference_service = InferenceService(model=model, preprocessor=preprocessor, threshold=threshold)

@app.get("/")
def root():
    return {"message": "Customer Churn API is running"}

# class ChurnRequest(BaseModel):
#     data: dict
class ChurnRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(request: ChurnRequest):
    return inference_service.predict(request.data)

# @app.post("/predict")
# def predict(request: ChurnRequest):
#     # 1️⃣ Convert input dict → DataFrame
#     import pandas as pd
#     df = pd.DataFrame([request.data])

#     # 2️⃣ Apply SAME preprocessor
#     X = preprocessor.transform(df)

#     # 3️⃣ Predict probability
#     # proba = model.predict_proba(X)[:, 1][0]
#     proba = float(model.predict(X)[0])

#     # 4️⃣ Apply tuned threshold
#     prediction = int(proba >= threshold)

#     return {
#         "churn_probability": round(float(proba), 4),
#         "threshold": threshold,
#         "prediction": prediction,
#     }

@app.get("/health")
def health():
    return {"status": "healthy"}


# from fastapi import FastAPI, HTTPException
# from contextlib import asynccontextmanager

# from src.serving.schemas import ChurnRequest, ChurnResponse
# from src.serving.inference_service import InferenceService
# from src.serving.model_loader import load_production_model

# # -------------------------------------------------
# # App lifecycle (startup / shutdown)
# # -------------------------------------------------

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Load model once at startup.
#     """
#     try:
#         model, threshold = load_production_model()
#         app.state.inference_service = InferenceService(
#             model=model,
#             threshold=threshold
#         )
#         print("✅ Model loaded successfully at startup")
#     except Exception as e:
#         print("❌ Failed to load model at startup:", e)
#         raise e

#     yield

#     print("🛑 Application shutdown")


# # -------------------------------------------------
# # FastAPI app
# # -------------------------------------------------

# app = FastAPI(
#     title="Customer Churn Prediction API",
#     version="1.0.0",
#     lifespan=lifespan
# )


# # -------------------------------------------------
# # Health check
# # -------------------------------------------------

# @app.get("/health")
# def health_check():
#     return {
#         "status": "ok",
#         "model_loaded": True
#     }


# # -------------------------------------------------
# # Prediction endpoint
# # -------------------------------------------------

# @app.post("/predict", response_model=ChurnResponse)
# def predict(request: ChurnRequest):
#     try:
#         service: InferenceService = app.state.inference_service
#         result = service.predict(request.features)
#         return result

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Inference failed: {str(e)}"
#         )