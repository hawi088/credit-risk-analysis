import os 
import sys 
import pandas as pd 
import joblib 
from fastapi import FastAPI, HTTPException 
from typing import List 
import warnings 
warnings.filterwarnings('ignore') 
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse 
from data_processing import process_raw_data 
 
app = FastAPI(title="Credit Risk Model API", version="1.0.0") 
model = None 
 
def load_model(): 
    global model 
    if os.path.exists('best_model.pkl'): 
        model = joblib.load('best_model.pkl') 
        return True 
    return False 
 
@app.on_event("startup") 
async def startup(): 
    load_model() 
 
@app.get("/health", response_model=HealthResponse) 
async def health(): 
    return HealthResponse(status="healthy", model_loaded=model is not None, version="1.0.0") 
 
@app.get("/", response_model=HealthResponse) 
async def root(): 
    return await health() 
 
@app.post("/predict", response_model=List[PredictionResponse]) 
async def predict(request: PredictionRequest): 
    if model is None: 
        raise HTTPException(503, "Model not loaded") 
    df = pd.DataFrame([t.dict() for t in request.transactions]) 
    features = process_raw_data(df) 
    if 'is_high_risk' in features.columns: 
        features = features.drop('is_high_risk', axis=1) 
    probs = model.predict_proba(features)[:, 1] 
    customers = df['CustomerId'].unique() 
    responses = [] 
    for i, cust in enumerate(customers): 
        p = float(probs[i]) 
        score = (1 - p) * 1000 
        responses.append(PredictionResponse( 
            customer_id=str(cust), 
            risk_probability=round(p,4), 
            risk_score=round(score,2), 
            risk_level=level, 
            is_high_risk=p 
        )) 
    return responses 
