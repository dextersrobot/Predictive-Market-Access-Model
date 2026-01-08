"""
Market Access Prediction Model

This module trains and evaluates machine learning models to predict
coverage barriers for new drug approvals.

Models:
1. Logistic Regression (baseline, interpretable)
2. Random Forest (primary model)
3. XGBoost (high performance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back to sklearn if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available, using GradientBoosting instead")


class MarketAccessPredictor:
    """Train and evaluate market access prediction models."""
    
    def __init__(
        self, 
        features_dir: str = "data/features",
        models_dir: str = "models"
    ):
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def load_training_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load training data and feature metadata."""
        # Load training data
        train_path = self.features_dir / "training_data.csv"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Run feature engineering first."
            )
        
        df = pd.read_csv(train_path)
        
        # Load feature metadata
        meta_path = self.features_dir / "feature_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        
        feature_columns = metadata["feature_columns"]
        
        print(f"Loaded {len(df)} records with {len(feature_columns)} features")
        return df, feature_columns
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple:
        """Prepare data for model training."""
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df["target_high_barrier"].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")
        print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()
    
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train logistic regression baseline model."""
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Fit final model
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """Train random forest model."""
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Fit final model
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series
    ):
        """Train gradient boosting model (XGBoost or sklearn)."""
        print("\nTraining Gradient Boosting...")
        
        if XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Fit final model
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        return model
    
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """Evaluate model performance."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {results['accuracy']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1 Score:  {results['f1']:.3f}")
        print(f"  ROC AUC:   {results['roc_auc']:.3f}")
        
        return results
    
    def get_feature_importance(
        self, 
        model, 
        feature_names: List[str],
        model_name: str
    ) -> pd.DataFrame:
        """Extract feature importance from model."""
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n{model_name} - Top 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_models(self):
        """Save trained models and results."""
        
        # Save models
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save results
        results_path = self.models_dir / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nAll models saved to {self.models_dir}")
    
    def run_training_pipeline(self) -> Dict:
        """Run the full model training pipeline."""
        print("="*60)
        print("MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        print("\n[1/5] Loading training data...")
        df, feature_columns = self.load_training_data()
        
        # Prepare data
        print("\n[2/5] Preparing data...")
        X_train, X_test, y_train, y_test, features = self.prepare_data(df, feature_columns)
        
        # Train models
        print("\n[3/5] Training models...")
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        print("\n[4/5] Evaluating models...")
        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)
            self.get_feature_importance(model, features, name)
        
        # Save models
        print("\n[5/5] Saving models...")
        self.save_models()
        
        return self.results


def main():
    """Main entry point."""
    predictor = MarketAccessPredictor()
    results = predictor.run_training_pipeline()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\nBest model: {best_model[0]} (AUC: {best_model[1]['roc_auc']:.3f})")
    
    print("\nNext steps:")
    print("1. Review feature importance to understand drivers")
    print("2. Run: python src/visualization/dashboard.py")
    print("3. Use model for predictions on new drugs")
    
    return results


if __name__ == "__main__":
    main()
