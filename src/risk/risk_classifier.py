import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class CostSensitiveRiskModel:
    def __init__(self, risk_threshold=65, model_path="models_store/cost_sensitive_risk.pkl"):
        self.risk_threshold = risk_threshold
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}

    def _engineer_temporal_features(self, df):
        """Addresses the 'Temporal' requirement by simulating time-phases."""
        X = df.copy()
        sleep = X['Sleep_Hours'].replace(0, 0.1) # Prevent div-by-zero
        
        # Phase 1: Early Term
        X['Phase1_Engagement'] = X['Attendance'] * (X['Motivation_Level'].map({'Low': 0.5, 'Medium': 1.0, 'High': 1.5}).fillna(1.0))
        # Phase 2: Mid Term
        X['Phase2_Consistency'] = X['Previous_Scores'] + (X['Tutoring_Sessions'] * 2)
        # Phase 3: Late Term
        X['Phase3_Fatigue'] = (X['Hours_Studied'] / sleep) * X['Physical_Activity'].apply(lambda x: 1 if x < 2 else 1.2)
        return X

    def preprocess(self, X, is_training=True):
        X_temporal = self._engineer_temporal_features(X)
        cat_cols = X_temporal.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            if is_training:
                le = LabelEncoder()
                X_temporal[col] = le.fit_transform(X_temporal[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    X_temporal[col] = X_temporal[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else -1
                    )
        return X_temporal

    def train(self, data_path='data/raw/StudentPerformanceFactors.csv'):
        """Addresses 'Imbalance' and 'Cost-Sensitive Learning'."""
        print("Loading dataset for Risk Modeling...")
        df = pd.read_csv(data_path)
        
        # Define minority class
        df['At_Risk'] = (df['Exam_Score'] < self.risk_threshold).astype(int)
        X = df.drop(columns=['Exam_Score', 'At_Risk'])
        y = df['At_Risk']

        X_encoded = self.preprocess(X, is_training=True)

        # 1. Class Imbalance (SMOTE)
        print("Applying SMOTE to balance at-risk minority class...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_encoded, y)

        # 2. Cost-Sensitive Learning Matrix
        imbalance_ratio = len(y_res[y_res == 0]) / len(y_res[y_res == 1])
        cost_penalty = imbalance_ratio * 3.0 

        self.model = xgb.XGBClassifier(
            scale_pos_weight=cost_penalty,
            eval_metric='aucpr',
            random_state=42
        )
        
        print("Training Cost-Sensitive XGBoost Classifier...")
        self.model.fit(X_res, y_res)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'encoders': self.label_encoders}, self.model_path)
        print("Model saved successfully.")

    def predict_risk(self, student_dict):
        """Addresses 'Prediction Uncertainty' and aligns column orders."""
        if not self.model:
            artifacts = joblib.load(self.model_path)
            self.model, self.label_encoders = artifacts['model'], artifacts['encoders']

        df_in = pd.DataFrame([student_dict])
        X_proc = self.preprocess(df_in, is_training=False)
        
        # Enforce exact column order and matching schema from training
        if hasattr(self.model, "feature_names_in_"):
            for col in self.model.feature_names_in_:
                if col not in X_proc.columns:
                    X_proc[col] = 0
            X_proc = X_proc[self.model.feature_names_in_]
        
        probs = self.model.predict_proba(X_proc)[0]
        safe_p, risk_p = probs[0], probs[1]
        
        # Calculate Shannon Entropy for Prediction Uncertainty
        entropy = - (safe_p * np.log2(safe_p + 1e-9) + risk_p * np.log2(risk_p + 1e-9))
        
        is_uncertain = bool(entropy > 0.85)
        is_at_risk = bool(risk_p > 0.5)

        return {
            "risk_flag": is_at_risk,
            "risk_probability": round(float(risk_p * 100), 2),
            "uncertainty_score": round(float(entropy), 4),
            "is_uncertain": is_uncertain,
            "status": "Review Required (High Uncertainty)" if is_uncertain else ("At Risk" if is_at_risk else "Safe")
        }