import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    roc_curve
)
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import os

class AbandonmentPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.income_bins = None
        
    def engineer_features(self, df):
        """Create derived features"""
        df = df.copy()
        
        # Cost-based features
        df['oop_to_income_ratio'] = df['oop_cost'] / (df['median_income'] / 12)
        df['is_high_cost'] = (df['oop_cost'] > 500).astype(int)
        df['cost_per_mile'] = df['oop_cost'] / (df['distance_to_pharmacy'] + 1)
        
        # Income-based features
        df['is_low_income'] = (df['median_income'] < 50000).astype(int)
        
        # Use fixed bins instead of qcut
        if self.income_bins is None:
            df['income_percentile'] = pd.cut(
                df['median_income'], 
                bins=[0, 40000, 60000, 80000, float('inf')], 
                labels=[1, 2, 3, 4]
            )
        else:
            df['income_percentile'] = pd.cut(
                df['median_income'],
                bins=self.income_bins,
                labels=[1, 2, 3, 4]
            )
        
        # Access barriers
        df['is_pharmacy_desert'] = (df['distance_to_pharmacy'] > 20).astype(int)
        df['access_barrier_count'] = (
            df['pa_required'].astype(int) + 
            df['is_pharmacy_desert'] + 
            (df['distance_to_pharmacy'] > 30).astype(int)
        )
        
        # Interaction features
        df['high_cost_low_income'] = (
            (df['oop_cost'] > 500) & (df['median_income'] < 50000)
        ).astype(int)
        
        df['uninsured_high_cost'] = (
            (df['insurance_type'] == 'Uninsured') & (df['oop_cost'] > 1000)
        ).astype(int)
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], 
                                 labels=['young', 'middle', 'senior', 'elderly'])
        
        return df
    
    def prepare_features(self, df, fit=False):
        """Prepare features for modeling"""
        df = self.engineer_features(df)
        
        # Features to use
        numeric_features = [
            'age', 'oop_cost', 'distance_to_pharmacy', 
            'prior_abandonment_count', 'oop_to_income_ratio',
            'median_income', 'drug_cost', 'access_barrier_count'
        ]
        
        categorical_features = [
            'insurance_type', 'therapeutic_area', 'age_group'
        ]
        
        binary_features = [
            'pa_required', 'is_high_cost', 'is_low_income',
            'is_pharmacy_desert', 'high_cost_low_income', 'uninsured_high_cost'
        ]
        
        # Encode categorical variables
        X = df.copy()
        for col in categorical_features:
            if fit:
                le = LabelEncoder()
                X[col + '_encoded'] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                X[col + '_encoded'] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Add income percentile as numeric
        X['income_percentile_num'] = X['income_percentile'].astype(int)
        
        # Select final features
        feature_cols = (
            numeric_features + 
            binary_features + 
            ['income_percentile_num'] +
            [col + '_encoded' for col in categorical_features]
        )
        
        X_final = X[feature_cols]
        
        # Scale numeric features
        numeric_to_scale = numeric_features + ['income_percentile_num']
        if fit:
            X_final[numeric_to_scale] = self.scaler.fit_transform(X_final[numeric_to_scale])
            self.feature_names = feature_cols
            self.income_bins = [0, 40000, 60000, 80000, float('inf')]
        else:
            X_final[numeric_to_scale] = self.scaler.transform(X_final[numeric_to_scale])
        
        return X_final
    
    def train(self, df):
        """Train the abandonment prediction model"""
        print("Preparing features...")
        X = self.prepare_features(df, fit=True)
        y = df['abandoned']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training set: {} samples".format(len(X_train)))
        print("Test set: {} samples".format(len(X_test)))
        print("Abandonment rate - Train: {:.2%}, Test: {:.2%}".format(y_train.mean(), y_test.mean()))
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n=== Model Performance ===")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        print("AUC-ROC: {:.3f}".format(auc))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Filled', 'Abandoned']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Cross-validation
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        print("CV AUC: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std()))
        
        return {
            'auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def predict(self, df):
        """Predict abandonment probability"""
        X = self.prepare_features(df, fit=False)
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities
    
    def save(self, path='models/abandonment_model.joblib'):
        """Save model and preprocessing objects"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'income_bins': self.income_bins
        }, path)
        print("\nModel saved to {}".format(path))
    
    @classmethod
    def load(cls, path='models/abandonment_model.joblib'):
        """Load saved model"""
        predictor = cls()
        saved_objects = joblib.load(path)
        predictor.model = saved_objects['model']
        predictor.scaler = saved_objects['scaler']
        predictor.label_encoders = saved_objects['label_encoders']
        predictor.feature_names = saved_objects['feature_names']
        predictor.income_bins = saved_objects.get('income_bins', [0, 40000, 60000, 80000, float('inf')])
        return predictor


if __name__ == '__main__':
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/synthetic_patients.csv')
    
    # Train model
    predictor = AbandonmentPredictor()
    metrics = predictor.train(df)
    
    # Save model
    predictor.save()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Model AUC: {:.3f}".format(metrics['auc']))
    print("CV AUC: {:.3f} (+/- {:.3f})".format(metrics['cv_auc_mean'], metrics['cv_auc_std']))
