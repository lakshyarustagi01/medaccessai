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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'abandonment_model.joblib')
print(f"Loading model from: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
predictor = AbandonmentPredictor.load(model_path)
recommender = InterventionRecommender()

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

@app.get("/zip-income/{zip_code}")
def get_zip_income(zip_code: str):
    state_income = {
        'AL': 52035, 'AK': 77640, 'AZ': 62055, 'AR': 49475, 'CA': 78672,
        'CO': 77127, 'CT': 78833, 'DE': 70176, 'FL': 59227, 'GA': 61980,
        'HI': 83102, 'ID': 60999, 'IL': 68428, 'IN': 57603, 'IA': 61691,
        'KS': 61091, 'KY': 52295, 'LA': 51073, 'ME': 59489, 'MD': 86738,
        'MA': 84385, 'MI': 59584, 'MN': 74593, 'MS': 46511, 'MO': 57409,
        'MT': 57153, 'NE': 63229, 'NV': 63276, 'NH': 81160, 'NJ': 85751,
        'NM': 51945, 'NY': 72108, 'NC': 57341, 'ND': 65315, 'OH': 58642,
        'OK': 54449, 'OR': 67058, 'PA': 63463, 'RI': 71169, 'SC': 56227,
        'SD': 59533, 'TN': 56071, 'TX': 64034, 'UT': 75780, 'VT': 63001,
        'VA': 76456, 'WA': 78687, 'WV': 48850, 'WI': 64168, 'WY': 65003
    }
    
    try:
        z = int(zip_code[:3])
        s = None
        
        if 350 <= z <= 369: s = 'AL'
        elif 995 <= z <= 999: s = 'AK'
        elif 850 <= z <= 865: s = 'AZ'
        elif 716 <= z <= 729: s = 'AR'
        elif 900 <= z <= 961: s = 'CA'
        elif 800 <= z <= 816: s = 'CO'
        elif 60 <= z <= 69: s = 'CT'
        elif 197 <= z <= 199: s = 'DE'
        elif 320 <= z <= 349: s = 'FL'
        elif 300 <= z <= 319: s = 'GA'
        elif 967 <= z <= 968: s = 'HI'
        elif 832 <= z <= 838: s = 'ID'
        elif 600 <= z <= 629: s = 'IL'
        elif 460 <= z <= 479: s = 'IN'
        elif 500 <= z <= 528: s = 'IA'
        elif 660 <= z <= 679: s = 'KS'
        elif 400 <= z <= 427: s = 'KY'
        elif 700 <= z <= 714: s = 'LA'
        elif 39 <= z <= 49: s = 'ME'
        elif 206 <= z <= 219: s = 'MD'
        elif 10 <= z <= 27: s = 'MA'
        elif 480 <= z <= 499: s = 'MI'
        elif 550 <= z <= 567: s = 'MN'
        elif 386 <= z <= 397: s = 'MS'
        elif 630 <= z <= 658: s = 'MO'
        elif 590 <= z <= 599: s = 'MT'
        elif 680 <= z <= 693: s = 'NE'
        elif 889 <= z <= 898: s = 'NV'
        elif 30 <= z <= 38: s = 'NH'
        elif 70 <= z <= 89: s = 'NJ'
        elif 870 <= z <= 884: s = 'NM'
        elif (5 <= z <= 9) or (100 <= z <= 149): s = 'NY'
        elif 270 <= z <= 289: s = 'NC'
        elif 580 <= z <= 588: s = 'ND'
        elif 430 <= z <= 458: s = 'OH'
        elif 730 <= z <= 749: s = 'OK'
        elif 970 <= z <= 979: s = 'OR'
        elif 150 <= z <= 196: s = 'PA'
        elif 28 <= z <= 29: s = 'RI'
        elif 290 <= z <= 299: s = 'SC'
        elif 570 <= z <= 577: s = 'SD'
        elif 370 <= z <= 385: s = 'TN'
        elif (750 <= z <= 799) or (885 <= z <= 888): s = 'TX'
        elif 840 <= z <= 847: s = 'UT'
        elif 50 <= z <= 59: s = 'VT'
        elif (201 <= z <= 205) or (220 <= z <= 246): s = 'VA'
        elif 980 <= z <= 994: s = 'WA'
        elif 247 <= z <= 268: s = 'WV'
        elif 530 <= z <= 549: s = 'WI'
        elif 820 <= z <= 831: s = 'WY'
        
        if s and s in state_income:
            return {"zip_code": zip_code, "state": s, "median_income": state_income[s]}
        return {"zip_code": zip_code, "median_income": 65000}
    except:
        return {"zip_code": zip_code, "median_income": 65000}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
