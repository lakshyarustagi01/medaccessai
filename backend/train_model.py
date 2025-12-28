import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class AbandonmentPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic patient data with new high-impact features"""
        np.random.seed(42)
        
        data = {
            # Original features
            'age': np.random.randint(18, 90, n_samples),
            'zip_code': np.random.choice(['78701', '90210', '10001', '60601', '30301'], n_samples),
            'median_income': np.random.normal(65000, 25000, n_samples).clip(20000, 200000),
            'insurance_type': np.random.choice(['Commercial', 'Medicare', 'Medicaid', 'Uninsured'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
            'drug_cost': np.random.choice([3000, 5000, 7000, 12000, 16000, 20000], n_samples),
            'oop_cost': np.random.uniform(0, 5000, n_samples),
            'distance_to_pharmacy': np.random.exponential(10, n_samples).clip(0, 100),
            'pa_required': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
            'prior_abandonment_count': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            
            # NEW HIGH-IMPACT FEATURES
            'deductible_met': np.random.choice([True, False], n_samples, p=[0.35, 0.65]),
            'deductible_remaining': np.random.uniform(0, 6000, n_samples),
            'prescription_month': np.random.randint(1, 13, n_samples),
            'is_refill': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
            'days_since_prescription': np.random.randint(0, 30, n_samples),
            'administration_route': np.random.choice(['oral', 'injection', 'infusion'], n_samples, p=[0.5, 0.3, 0.2]),
            'specialty_pharmacy_required': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
            'lives_alone': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'primary_language': np.random.choice(['English', 'Spanish', 'Other'], n_samples, p=[0.75, 0.15, 0.1]),
            'has_caregiver': np.random.choice([True, False], n_samples, p=[0.4, 0.6])
        }
        
        df = pd.DataFrame(data)
        
        # Set deductible_remaining to 0 if deductible is met
        df.loc[df['deductible_met'] == True, 'deductible_remaining'] = 0
        
        # Generate abandonment based on realistic logic
        abandonment_prob = np.zeros(n_samples)
        
        # Base rate by insurance
        insurance_base = {'Commercial': 0.15, 'Medicare': 0.20, 'Medicaid': 0.25, 'Uninsured': 0.50}
        for ins_type, base_prob in insurance_base.items():
            mask = df['insurance_type'] == ins_type
            abandonment_prob[mask] += base_prob
        
        # OOP cost impact (BIGGEST FACTOR)
        oop_to_income_ratio = df['oop_cost'] / (df['median_income'] / 12)
        abandonment_prob += oop_to_income_ratio * 0.8  # Major impact
        
        # DEDUCTIBLE STATUS (HUGE IMPACT)
        deductible_not_met = ~df['deductible_met']
        abandonment_prob += deductible_not_met * 0.25  # Big penalty if deductible not met
        
        # Deductible remaining impact
        deductible_ratio = df['deductible_remaining'] / 6000  # Normalized
        abandonment_prob += deductible_ratio * 0.15
        
        # PRESCRIPTION MONTH (January/February = deductible reset)
        is_jan_feb = df['prescription_month'].isin([1, 2])
        abandonment_prob += is_jan_feb * 0.20  # Major spike in Jan/Feb
        
        # IS REFILL (existing patients less likely to abandon)
        is_new_patient = ~df['is_refill']
        abandonment_prob += is_new_patient * 0.15
        
        # Days since prescription (urgency decay)
        days_factor = (df['days_since_prescription'] / 30) * 0.10
        abandonment_prob += days_factor
        
        # Administration route (complex = more abandonment)
        route_penalty = {'oral': 0, 'injection': 0.05, 'infusion': 0.10}
        for route, penalty in route_penalty.items():
            mask = df['administration_route'] == route
            abandonment_prob[mask] += penalty
        
        # Social support factors
        abandonment_prob += df['lives_alone'] * 0.08
        abandonment_prob += ~df['has_caregiver'] * 0.06
        
        # Language barrier
        language_penalty = {'English': 0, 'Spanish': 0.07, 'Other': 0.10}
        for lang, penalty in language_penalty.items():
            mask = df['primary_language'] == lang
            abandonment_prob[mask] += penalty
        
        # Original factors
        abandonment_prob += (df['distance_to_pharmacy'] / 50) * 0.1
        abandonment_prob += df['pa_required'] * 0.12
        abandonment_prob += df['prior_abandonment_count'] * 0.15
        
        # Clip to valid probability range
        abandonment_prob = np.clip(abandonment_prob, 0, 1)
        
        # Generate actual abandonment with some noise
        df['abandoned'] = (np.random.random(n_samples) < abandonment_prob).astype(int)
        
        return df
    
    def engineer_features(self, df):
        """Create derived features"""
        df = df.copy()
        
        # Original engineered features
        df['oop_to_income_ratio'] = df['oop_cost'] / (df['median_income'] / 12)
        df['cost_burden_level'] = pd.cut(df['oop_to_income_ratio'], 
                                         bins=[0, 0.1, 0.3, 0.5, float('inf')],
                                         labels=['low', 'medium', 'high', 'critical'])
        df['high_cost_drug'] = (df['drug_cost'] > 10000).astype(int)
        df['far_from_pharmacy'] = (df['distance_to_pharmacy'] > 20).astype(int)
        df['low_income_area'] = (df['median_income'] < 50000).astype(int)
        
        # NEW ENGINEERED FEATURES
        df['deductible_factor'] = df['deductible_met'].apply(lambda x: 0.3 if x else 1.5)
        df['is_january_february'] = df['prescription_month'].isin([1, 2]).astype(int)
        df['is_new_patient'] = (~df['is_refill']).astype(int)
        df['prescription_aged'] = (df['days_since_prescription'] > 7).astype(int)
        df['complex_administration'] = df['administration_route'].isin(['injection', 'infusion']).astype(int)
        df['has_support'] = (~df['lives_alone'] | df['has_caregiver']).astype(int)
        df['language_barrier'] = (df['primary_language'] != 'English').astype(int)
        
        # Interaction features
        df['high_cost_no_deductible'] = ((df['oop_cost'] > 2000) & (~df['deductible_met'])).astype(int)
        df['new_patient_high_cost'] = (df['is_new_patient'] & (df['oop_cost'] > 1500)).astype(int)
        df['jan_feb_high_oop'] = (df['is_january_february'] & (df['oop_cost'] > 1000)).astype(int)
        
        # Encode categorical variables
        df['insurance_commercial'] = (df['insurance_type'] == 'Commercial').astype(int)
        df['insurance_medicare'] = (df['insurance_type'] == 'Medicare').astype(int)
        df['insurance_medicaid'] = (df['insurance_type'] == 'Medicaid').astype(int)
        df['insurance_uninsured'] = (df['insurance_type'] == 'Uninsured').astype(int)
        
        df['route_oral'] = (df['administration_route'] == 'oral').astype(int)
        df['route_injection'] = (df['administration_route'] == 'injection').astype(int)
        df['route_infusion'] = (df['administration_route'] == 'infusion').astype(int)
        
        df['lang_english'] = (df['primary_language'] == 'English').astype(int)
        df['lang_spanish'] = (df['primary_language'] == 'Spanish').astype(int)
        df['lang_other'] = (df['primary_language'] == 'Other').astype(int)
        
        # Drop original categorical columns for modeling
        df_encoded = df.drop(['insurance_type', 'zip_code', 'cost_burden_level', 
                             'administration_route', 'primary_language'], axis=1)
        
        return df_encoded
    
    def train(self, df):
        """Train XGBoost model"""
        df_features = self.engineer_features(df)
        
        X = df_features.drop('abandoned', axis=1)
        y = df_features['abandoned']
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Updated XGBoost parameters for better performance
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='auc'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE (WITH NEW FEATURES)")
        print(f"{'='*60}")
        print(f"Test AUC-ROC: {auc:.4f}")
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"{'='*60}\n")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("TOP 15 MOST IMPORTANT FEATURES:")
        print(feature_importance.head(15))
        
        return auc, cv_scores.mean()
    
    def predict(self, patient_data):
        """Predict abandonment probability"""
        if isinstance(patient_data, pd.DataFrame):
            df = patient_data
        else:
            df = pd.DataFrame([patient_data])
        
        df_features = self.engineer_features(df)
        X = df_features[self.feature_columns]
        
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities
    
    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_columns = model_data['feature_columns']
        return predictor

if __name__ == "__main__":
    print("Generating enhanced synthetic dataset...")
    predictor = AbandonmentPredictor()
    df = predictor.generate_synthetic_data(n_samples=10000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Abandonment rate: {df['abandoned'].mean():.2%}")
    
    print("\nTraining model with new high-impact features...")
    test_auc, cv_auc = predictor.train(df)
    
    print("\nSaving model...")
    predictor.save('models/abandonment_model.joblib')
    
    print("\n✅ Model training complete!")
    print(f"Expected improvement: 0.765 → {test_auc:.3f}")
