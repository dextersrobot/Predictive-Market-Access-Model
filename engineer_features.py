"""
Feature Engineering for Market Access Prediction

This module creates the final feature set for model training.
Features are organized into categories:
1. Regulatory features (approval pathway, designations)
2. Clinical features (therapeutic area, dosage form)
3. Economic features (pricing, market size proxies)
4. Temporal features (approval timing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Engineer features for market access prediction model."""
    
    # Known therapeutic areas (based on common FDA categorizations)
    THERAPEUTIC_AREAS = {
        "oncology": ["cancer", "tumor", "lymphoma", "leukemia", "melanoma", "carcinoma", "oncology"],
        "immunology": ["immune", "autoimmune", "rheumatoid", "psoriasis", "lupus", "crohn"],
        "neurology": ["neuro", "alzheimer", "parkinson", "epilepsy", "seizure", "migraine", "multiple sclerosis"],
        "cardiology": ["heart", "cardio", "hypertension", "cholesterol", "atrial", "angina"],
        "infectious": ["infection", "antibiotic", "antiviral", "hiv", "hepatitis", "bacterial"],
        "rare_disease": ["orphan", "rare", "enzyme replacement"],
        "diabetes": ["diabetes", "insulin", "glucose", "glp-1"],
        "respiratory": ["asthma", "copd", "pulmonary", "lung", "respiratory"],
    }
    
    def __init__(self, processed_dir: str = "data/processed", features_dir: str = "data/features"):
        self.processed_dir = Path(processed_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load the preprocessed merged data."""
        filepath = self.processed_dir / "merged_drug_data.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Processed data not found at {filepath}. "
                "Run preprocessing first: python src/preprocessing/merge_data.py"
            )
        
        df = pd.read_csv(filepath, parse_dates=["submission_status_date"])
        print(f"Loaded {len(df)} records from processed data")
        return df
    
    def infer_therapeutic_area(self, text: str) -> str:
        """Infer therapeutic area from drug name or description."""
        if not text or pd.isna(text):
            return "other"
        
        text_lower = text.lower()
        
        for area, keywords in self.THERAPEUTIC_AREAS.items():
            if any(kw in text_lower for kw in keywords):
                return area
        
        return "other"
    
    def create_regulatory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to regulatory pathway."""
        
        # Already have boolean features from preprocessing
        # Add combined regulatory advantage score
        regulatory_features = []
        
        if "is_priority_review" in df.columns:
            regulatory_features.append(df["is_priority_review"].astype(int))
        
        if "is_orphan" in df.columns:
            regulatory_features.append(df["is_orphan"].astype(int))
        
        if "is_new_molecular_entity" in df.columns:
            regulatory_features.append(df["is_new_molecular_entity"].astype(int))
        
        if regulatory_features:
            df["regulatory_advantage_score"] = sum(regulatory_features)
        else:
            df["regulatory_advantage_score"] = 0
        
        # Accelerated approval indicator (approximation based on submission class)
        if "submission_class_code_description" in df.columns:
            df["likely_accelerated"] = df["submission_class_code_description"].str.contains(
                "accelerated|breakthrough|fast track", case=False, na=False
            ).astype(int)
        else:
            df["likely_accelerated"] = 0
        
        return df
    
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to clinical characteristics."""
        
        # Infer therapeutic area
        text_cols = ["generic_name", "brand_name", "submission_class_code_description"]
        combined_text = ""
        for col in text_cols:
            if col in df.columns:
                combined_text = df[col].fillna("") + " "
        
        if "generic_name" in df.columns:
            df["therapeutic_area"] = df["generic_name"].apply(self.infer_therapeutic_area)
        else:
            df["therapeutic_area"] = "other"
        
        # Encode therapeutic area
        df["therapeutic_area_encoded"] = LabelEncoder().fit_transform(df["therapeutic_area"])
        
        # High-value therapeutic areas (oncology and rare diseases typically have faster coverage)
        df["is_high_value_area"] = df["therapeutic_area"].isin(
            ["oncology", "rare_disease", "immunology"]
        ).astype(int)
        
        # Dosage form complexity
        if "dosage_form" in df.columns:
            df["is_complex_dosage"] = df["dosage_form"].str.contains(
                "injection|infusion|implant|patch|inhalation", case=False, na=False
            ).astype(int)
            df["is_oral_simple"] = df["dosage_form"].str.contains(
                "tablet|capsule|oral", case=False, na=False
            ).astype(int)
        else:
            df["is_complex_dosage"] = 0
            df["is_oral_simple"] = 0
        
        # Biologic indicator (BLAs are typically more complex and expensive)
        df["is_biologic"] = df.get("is_bla", pd.Series([False]*len(df))).astype(int)
        
        return df
    
    def create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to economic factors."""
        
        # Price features
        if "launch_price" in df.columns:
            # Log transform price (handle zeros)
            df["log_launch_price"] = np.log1p(df["launch_price"].fillna(0))
            
            # Price tier (based on NADAC distribution)
            price_percentiles = df["launch_price"].quantile([0.25, 0.5, 0.75, 0.9])
            
            def price_tier(price):
                if pd.isna(price) or price == 0:
                    return 0  # Unknown
                elif price <= price_percentiles[0.25]:
                    return 1  # Low
                elif price <= price_percentiles[0.5]:
                    return 2  # Medium-low
                elif price <= price_percentiles[0.75]:
                    return 3  # Medium-high
                elif price <= price_percentiles[0.9]:
                    return 4  # High
                else:
                    return 5  # Very high
            
            df["price_tier"] = df["launch_price"].apply(price_tier)
        else:
            df["log_launch_price"] = 0
            df["price_tier"] = 0
        
        # Market size proxy (based on therapeutic area and dosage)
        # Oral drugs for common conditions = larger market
        df["market_size_proxy"] = 0
        
        if "therapeutic_area" in df.columns and "is_oral_simple" in df.columns:
            # Large markets: diabetes, cardiology with oral formulations
            large_market = (
                df["therapeutic_area"].isin(["diabetes", "cardiology", "respiratory"]) &
                (df["is_oral_simple"] == 1)
            )
            df.loc[large_market, "market_size_proxy"] = 2
            
            # Medium markets: oral drugs for other conditions
            medium_market = (df["is_oral_simple"] == 1) & ~large_market
            df.loc[medium_market, "market_size_proxy"] = 1
            
            # Small markets: injectable/complex + rare disease
            small_market = (df["is_complex_dosage"] == 1) | (df["therapeutic_area"] == "rare_disease")
            df.loc[small_market, "market_size_proxy"] = 0
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to approval timing."""
        
        if "submission_status_date" not in df.columns:
            df["approval_year"] = 2020
            df["approval_quarter"] = 1
            return df
        
        # Basic temporal features
        df["approval_year"] = df["submission_status_date"].dt.year
        df["approval_month"] = df["submission_status_date"].dt.month
        df["approval_quarter"] = df["submission_status_date"].dt.quarter
        
        # Cyclical encoding for month (sin/cos)
        df["month_sin"] = np.sin(2 * np.pi * df["approval_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["approval_month"] / 12)
        
        # Year since 2010 (normalized)
        df["years_since_2010"] = df["approval_year"] - 2010
        
        # Q4 approvals often face different payer dynamics (budget cycles)
        df["is_q4_approval"] = (df["approval_quarter"] == 4).astype(int)
        
        # Election year indicator (potential policy uncertainty)
        df["is_election_year"] = df["approval_year"].isin([2012, 2016, 2020, 2024]).astype(int)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Without actual coverage data, we create proxy targets based on
        characteristics that historically correlate with coverage outcomes.
        
        In production, you'd replace this with actual coverage dates from
        payer formulary data.
        """
        
        # Coverage difficulty score (proxy for actual coverage timeline)
        # Higher score = more likely to face coverage barriers
        
        df["coverage_difficulty_score"] = 0.0
        
        # Factors that typically increase coverage difficulty:
        
        # High price
        if "price_tier" in df.columns:
            df["coverage_difficulty_score"] += df["price_tier"] * 0.2
        
        # Complex/injectable formulation
        if "is_complex_dosage" in df.columns:
            df["coverage_difficulty_score"] += df["is_complex_dosage"] * 0.15
        
        # Biologic
        if "is_biologic" in df.columns:
            df["coverage_difficulty_score"] += df["is_biologic"] * 0.15
        
        # Factors that typically decrease coverage difficulty:
        
        # Priority review (suggests unmet need)
        if "is_priority_review" in df.columns:
            df["coverage_difficulty_score"] -= df["is_priority_review"].astype(int) * 0.1
        
        # Orphan drug (limited alternatives)
        if "is_orphan" in df.columns:
            df["coverage_difficulty_score"] -= df["is_orphan"].astype(int) * 0.15
        
        # High-value therapeutic area
        if "is_high_value_area" in df.columns:
            df["coverage_difficulty_score"] -= df["is_high_value_area"] * 0.1
        
        # Normalize to 0-1 range
        min_score = df["coverage_difficulty_score"].min()
        max_score = df["coverage_difficulty_score"].max()
        if max_score > min_score:
            df["coverage_difficulty_score"] = (
                (df["coverage_difficulty_score"] - min_score) / (max_score - min_score)
            )
        
        # Create binary target: high vs low coverage difficulty
        df["target_high_barrier"] = (df["coverage_difficulty_score"] > 0.5).astype(int)
        
        return df
    
    def select_model_features(self, df: pd.DataFrame) -> tuple:
        """Select features for model training."""
        
        # Define feature columns by category
        regulatory_features = [
            "is_priority_review", "is_orphan", "is_new_molecular_entity",
            "likely_accelerated", "regulatory_advantage_score"
        ]
        
        clinical_features = [
            "therapeutic_area_encoded", "is_high_value_area",
            "is_complex_dosage", "is_oral_simple", "is_biologic"
        ]
        
        economic_features = [
            "log_launch_price", "price_tier", "market_size_proxy"
        ]
        
        temporal_features = [
            "years_since_2010", "month_sin", "month_cos",
            "is_q4_approval", "is_election_year"
        ]
        
        # Combine all features
        all_features = regulatory_features + clinical_features + economic_features + temporal_features
        
        # Filter to features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Selected {len(available_features)} features for model:")
        for cat, feats in [
            ("Regulatory", regulatory_features),
            ("Clinical", clinical_features),
            ("Economic", economic_features),
            ("Temporal", temporal_features)
        ]:
            available = [f for f in feats if f in available_features]
            print(f"  {cat}: {len(available)} features")
        
        return available_features
    
    def run_feature_engineering(self) -> tuple:
        """Run the full feature engineering pipeline."""
        print("="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Load data
        print("\n[1/6] Loading processed data...")
        df = self.load_processed_data()
        
        # Create features
        print("\n[2/6] Creating regulatory features...")
        df = self.create_regulatory_features(df)
        
        print("\n[3/6] Creating clinical features...")
        df = self.create_clinical_features(df)
        
        print("\n[4/6] Creating economic features...")
        df = self.create_economic_features(df)
        
        print("\n[5/6] Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("\n[6/6] Creating target variable...")
        df = self.create_target_variable(df)
        
        # Select model features
        feature_columns = self.select_model_features(df)
        
        # Save engineered features
        output_path = self.features_dir / "engineered_features.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved engineered features to {output_path}")
        
        # Save feature list
        feature_meta = {
            "engineered_at": datetime.now().isoformat(),
            "total_records": len(df),
            "total_features": len(feature_columns),
            "feature_columns": feature_columns,
            "target_column": "target_high_barrier",
            "target_distribution": df["target_high_barrier"].value_counts().to_dict()
        }
        
        meta_path = self.features_dir / "feature_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(feature_meta, f, indent=2)
        
        # Create training-ready dataset
        train_data = df[feature_columns + ["target_high_barrier", "coverage_difficulty_score"]].copy()
        train_data = train_data.dropna(subset=feature_columns)
        
        train_path = self.features_dir / "training_data.csv"
        train_data.to_csv(train_path, index=False)
        print(f"Saved training data ({len(train_data)} records) to {train_path}")
        
        return df, feature_columns


def main():
    """Main entry point."""
    engineer = FeatureEngineer()
    df, features = engineer.run_feature_engineering()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(features)}")
    print(f"\nTarget distribution:")
    print(df["target_high_barrier"].value_counts())
    print(f"\nSample features:")
    print(df[features].describe())
    
    return df, features


if __name__ == "__main__":
    main()
