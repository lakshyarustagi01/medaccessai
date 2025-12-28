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
bashcd ~/medaccessai/backend/app

cat >> main.py << 'EOF'

# Add this new endpoint
@app.get("/zip-income/{zip_code}")
def get_zip_income(zip_code: str):
    """Get median income for a ZIP code"""
    try:
        # State-level median income data (2023 Census estimates)
        # In production, would use actual ZIP-level Census API
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
        
        # Simple ZIP to state mapping (first 3 digits)
        zip_prefix = zip_code[:3]
        zip_to_state = {
            '350-369': 'AL', '995-999': 'AK', '850-865': 'AZ', '716-729': 'AR',
            '900-961': 'CA', '800-816': 'CO', '060-069': 'CT', '197-199': 'DE',
            '320-349': 'FL', '300-319': 'GA', '967-968': 'HI', '832-838': 'ID',
            '600-629': 'IL', '460-479': 'IN', '500-528': 'IA', '660-679': 'KS',
            '400-427': 'KY', '700-714': 'LA', '039-049': 'ME', '206-219': 'MD',
            '010-027': 'MA', '480-499': 'MI', '550-567': 'MN', '386-397': 'MS',
            '630-658': 'MO', '590-599': 'MT', '680-693': 'NE', '889-898': 'NV',
            '030-038': 'NH', '070-089': 'NJ', '870-884': 'NM', '005-009,100-149': 'NY',
            '270-289': 'NC', '580-588': 'ND', '430-458': 'OH', '730-749': 'OK',
            '970-979': 'OR', '150-196': 'PA', '028-029': 'RI', '290-299': 'SC',
            '570-577': 'SD', '370-385': 'TN', '750-799,885-888': 'TX', '840-847': 'UT',
            '050-059': 'VT', '201-205,220-246': 'VA', '980-994': 'WA', '247-268': 'WV',
            '530-549': 'WI', '820-831': 'WY'
        }
        
        # Determine state from ZIP
        state = None
        zip_int = int(zip_prefix)
        if 350 <= zip_int <= 369: state = 'AL'
        elif 995 <= zip_int <= 999: state = 'AK'
        elif 850 <= zip_int <= 865: state = 'AZ'
        elif 716 <= zip_int <= 729: state = 'AR'
        elif 900 <= zip_int <= 961: state = 'CA'
        elif 800 <= zip_int <= 816: state = 'CO'
        elif 60 <= zip_int <= 69: state = 'CT'
        elif 197 <= zip_int <= 199: state = 'DE'
        elif 320 <= zip_int <= 349: state = 'FL'
        elif 300 <= zip_int <= 319: state = 'GA'
        elif 967 <= zip_int <= 968: state = 'HI'
        elif 832 <= zip_int <= 838: state = 'ID'
        elif 600 <= zip_int <= 629: state = 'IL'
        elif 460 <= zip_int <= 479: state = 'IN'
        elif 500 <= zip_int <= 528: state = 'IA'
        elif 660 <= zip_int <= 679: state = 'KS'
        elif 400 <= zip_int <= 427: state = 'KY'
        elif 700 <= zip_int <= 714: state = 'LA'
        elif 39 <= zip_int <= 49: state = 'ME'
        elif 206 <= zip_int <= 219: state = 'MD'
        elif 10 <= zip_int <= 27: state = 'MA'
        elif 480 <= zip_int <= 499: state = 'MI'
        elif 550 <= zip_int <= 567: state = 'MN'
        elif 386 <= zip_int <= 397: state = 'MS'
        elif 630 <= zip_int <= 658: state = 'MO'
        elif 590 <= zip_int <= 599: state = 'MT'
        elif 680 <= zip_int <= 693: state = 'NE'
        elif 889 <= zip_int <= 898: state = 'NV'
        elif 30 <= zip_int <= 38: state = 'NH'
        elif 70 <= zip_int <= 89: state = 'NJ'
        elif 870 <= zip_int <= 884: state = 'NM'
        elif (5 <= zip_int <= 9) or (100 <= zip_int <= 149): state = 'NY'
        elif 270 <= zip_int <= 289: state = 'NC'
        elif 580 <= zip_int <= 588: state = 'ND'
        elif 430 <= zip_int <= 458: state = 'OH'
        elif 730 <= zip_int <= 749: state = 'OK'
        elif 970 <= zip_int <= 979: state = 'OR'
        elif 150 <= zip_int <= 196: state = 'PA'
        elif 28 <= zip_int <= 29: state = 'RI'
        elif 290 <= zip_int <= 299: state = 'SC'
        elif 570 <= zip_int <= 577: state = 'SD'
        elif 370 <= zip_int <= 385: state = 'TN'
        elif (750 <= zip_int <= 799) or (885 <= zip_int <= 888): state = 'TX'
        elif 840 <= zip_int <= 847: state = 'UT'
        elif 50 <= zip_int <= 59: state = 'VT'
        elif (201 <= zip_int <= 205) or (220 <= zip_int <= 246): state = 'VA'
        elif 980 <= zip_int <= 994: state = 'WA'
        elif 247 <= zip_int <= 268: state = 'WV'
        elif 530 <= zip_int <= 549: state = 'WI'
        elif 820 <= zip_int <= 831: state = 'WY'
        
        if state and state in state_income:
            return {
                "zip_code": zip_code,
                "state": state,
                "median_income": state_income[state],
                "source": "2023 Census state-level estimate"
            }
        else:
            return {"zip_code": zip_code, "median_income": 65000, "source": "default"}
            
    except Exception as e:
        return {"zip_code": zip_code, "median_income": 65000, "source": "default"}
