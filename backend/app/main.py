from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import AbandonmentPredictor
from recommendations import InterventionRecommender

app = FastAPI(
    title="MedAccessAI API",
    description="Medication Abandonment Prediction & Intervention Recommendation API",
    version="0.1.0"
)

# More permissive CORS for mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Change to False for wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'abandonment_model.joblib')
print("Loading model from: {}".format(model_path))
print("File exists: {}".format(os.path.exists(model_path)))
predictor = AbandonmentPredictor.load(model_path)
recommender = InterventionRecommender()

# Request/Response models
class PredictionRequest(BaseModel):
    patient_id: Optional[str] = None
    age: int = Field(..., ge=18, le=120)
    zip_code: str
    median_income: float = Field(..., gt=0)
    insurance_type: str
    drug_name: str
    therapeutic_area: str
    drug_cost: float = Field(..., gt=0)
    oop_cost: float = Field(..., ge=0)
    distance_to_pharmacy: float = Field(..., ge=0)
    pa_required: bool
    prior_abandonment_count: int = Field(0, ge=0)
    prescription_date: Optional[str] = None

class Intervention(BaseModel):
    intervention_id: str
    name: str
    description: str
    priority: int
    estimated_cost_reduction: float
    new_oop_cost: float
    success_probability: float
    processing_time_days: int
    expected_impact: float
    action_items: List[str]

class PredictionResponse(BaseModel):
    patient_id: Optional[str]
    abandonment_risk: float
    risk_level: str
    risk_factors: List[str]
    recommendations: List[Intervention]

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "MedAccessAI API",
        "version": "0.1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_abandonment(request: PredictionRequest):
    try:
        import pandas as pd
        patient_df = pd.DataFrame([request.model_dump()])
        
        abandonment_prob = predictor.predict(patient_df)[0]
        
        if abandonment_prob < 0.3:
            risk_level = "LOW"
        elif abandonment_prob < 0.5:
            risk_level = "MEDIUM"
        elif abandonment_prob < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        risk_factors = []
        
        if request.oop_cost > 1000:
            risk_factors.append("High out-of-pocket cost (${:,.0f})".format(request.oop_cost))
        
        oop_to_income_ratio = (request.oop_cost / (request.median_income / 12))
        if oop_to_income_ratio > 0.3:
            risk_factors.append("OOP-to-income ratio exceeds 30% ({:.0%})".format(oop_to_income_ratio))
        
        if request.insurance_type == "Uninsured":
            risk_factors.append("Patient is uninsured")
        
        if request.pa_required:
            risk_factors.append("Prior authorization required")
        
        if request.distance_to_pharmacy > 20:
            risk_factors.append("Far from specialty pharmacy ({:.1f} miles)".format(request.distance_to_pharmacy))
        
        if request.prior_abandonment_count > 0:
            risk_factors.append("History of abandonment ({} prior)".format(request.prior_abandonment_count))
        
        if request.median_income < 50000:
            risk_factors.append("Low-income area (${:,.0f} median)".format(request.median_income))
        
        recommendations = recommender.recommend(
            request.model_dump(),
            abandonment_prob
        )
        
        return PredictionResponse(
            patient_id=request.patient_id,
            abandonment_risk=round(abandonment_prob, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        import traceback
        print("Error details:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed: {}".format(str(e)))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "recommender_loaded": recommender is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
