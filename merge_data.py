"""
Data Preprocessing and Merging Pipeline

This module merges the collected data sources into a unified dataset
for feature engineering and model training.

Merge Strategy:
1. FDA Approvals as the base dataset (each drug approval is one row)
2. Left join orphan drug status
3. Left join CMS coverage data (match on drug name/indication)
4. Left join NADAC pricing (match on NDC where possible)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
from difflib import SequenceMatcher


class DataPreprocessor:
    """Preprocess and merge collected data sources."""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fda_approvals(self) -> pd.DataFrame:
        """Load and clean FDA approvals data."""
        filepath = self.raw_dir / "fda_approvals.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"FDA approvals not found at {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=["submission_status_date"])
        
        # Clean and standardize
        df["generic_name_clean"] = df["generic_name"].fillna("").str.lower().str.strip()
        df["brand_name_clean"] = df["brand_name"].fillna("").str.lower().str.strip()
        
        # Filter to original NDAs and BLAs (not supplements)
        original_approvals = df[
            (df["submission_type"].isin(["ORIG", "NDA", "BLA"])) |
            (df["submission_number"] == "1")
        ].copy()
        
        print(f"Loaded {len(df)} FDA records, filtered to {len(original_approvals)} original approvals")
        return original_approvals
    
    def load_orphan_drugs(self) -> pd.DataFrame:
        """Load orphan drug designations."""
        filepath = self.raw_dir / "orphan_drugs_lookup.csv"
        if not filepath.exists():
            print("Warning: Orphan drug data not found, skipping...")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df["drug_name_std"] = df["generic_name"].fillna("").str.lower().str.strip()
        
        print(f"Loaded {len(df)} orphan drug records")
        return df
    
    def load_cms_coverage(self) -> pd.DataFrame:
        """Load CMS NCD coverage data."""
        filepath = self.raw_dir / "cms_ncds.csv"
        if not filepath.exists():
            print("Warning: CMS coverage data not found, skipping...")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, parse_dates=["effective_date", "implementation_date"])
        
        # Filter to current NCDs
        df = df[df["is_current"] == True].copy()
        
        print(f"Loaded {len(df)} current CMS NCDs")
        return df
    
    def load_nadac_pricing(self) -> pd.DataFrame:
        """Load NADAC pricing data."""
        filepath = self.raw_dir / "nadac_prices.csv"
        if not filepath.exists():
            print("Warning: NADAC pricing data not found, skipping...")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        # Parse date columns if they exist
        for col in ["as_of_date", "effective_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Get latest price per NDC if we have the data
        if "as_of_date" in df.columns and "ndc" in df.columns:
            df = df.sort_values("as_of_date", ascending=False)
            latest_prices = df.groupby("ndc").first().reset_index()
        else:
            latest_prices = df
        
        print(f"Loaded {len(latest_prices)} unique NDC prices from NADAC")
        return latest_prices
    
    def fuzzy_match_drug_name(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two drug names are similar enough to match."""
        if not name1 or not name2:
            return False
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() >= threshold
    
    def merge_orphan_status(self, fda_df: pd.DataFrame, orphan_df: pd.DataFrame) -> pd.DataFrame:
        """Merge orphan drug status with FDA approvals."""
        if orphan_df.empty:
            fda_df["is_orphan"] = False
            return fda_df
        
        # Create lookup set for faster matching
        orphan_names = set(orphan_df["drug_name_std"].dropna().unique())
        
        # Simple exact match first
        fda_df["is_orphan"] = fda_df["generic_name_clean"].isin(orphan_names)
        
        # Count matches
        matched = fda_df["is_orphan"].sum()
        print(f"Matched {matched} drugs to orphan status ({matched/len(fda_df)*100:.1f}%)")
        
        return fda_df
    
    def add_coverage_features(self, df: pd.DataFrame, cms_df: pd.DataFrame) -> pd.DataFrame:
        """Add coverage-related features from CMS data."""
        if cms_df.empty:
            df["has_ncd"] = False
            df["ncd_category"] = None
            return df
        
        # This is a simplified matching - in production you'd want more sophisticated matching
        # based on therapeutic category and indication
        
        # For now, just add placeholder columns
        df["has_ncd"] = False
        df["ncd_category"] = None
        df["ncd_effective_date"] = None
        
        # Note: Proper NCD matching requires manual review or NLP
        # since NCDs are by procedure/indication, not by drug name
        print("Note: NCD matching requires manual curation for accuracy")
        
        return df
    
    def add_pricing_features(self, df: pd.DataFrame, nadac_df: pd.DataFrame) -> pd.DataFrame:
        """Add pricing features from NADAC data."""
        if nadac_df.empty:
            df["has_nadac_price"] = False
            df["launch_price"] = None
            df["is_brand"] = False
            return df
        
        # Check if we have the expected columns
        if "ndc_description" not in nadac_df.columns or "nadac_per_unit" not in nadac_df.columns:
            print("Warning: NADAC data missing expected columns, using defaults")
            df["has_nadac_price"] = False
            df["launch_price"] = None
            df["is_brand"] = False
            return df
        
        # Create drug name lookup from NADAC descriptions
        nadac_df = nadac_df.copy()
        nadac_df["drug_name_extracted"] = (
            nadac_df["ndc_description"]
            .fillna("")
            .str.lower()
            .str.split()
            .str[0]  # First word is usually the drug name
        )
        
        # Add is_brand if not present
        if "is_brand" not in nadac_df.columns:
            if "classification_for_rate_setting" in nadac_df.columns:
                nadac_df["is_brand"] = nadac_df["classification_for_rate_setting"].str.upper() == "B"
            else:
                nadac_df["is_brand"] = False
        
        # Get average price by drug name
        price_by_name = nadac_df.groupby("drug_name_extracted").agg({
            "nadac_per_unit": "mean",
            "is_brand": "first"
        }).reset_index()
        
        # Merge on generic name
        df["drug_name_match"] = df["generic_name_clean"].str.split().str[0]
        
        df = df.merge(
            price_by_name,
            left_on="drug_name_match",
            right_on="drug_name_extracted",
            how="left"
        )
        
        df["has_nadac_price"] = df["nadac_per_unit"].notna()
        df["launch_price"] = df["nadac_per_unit"]
        
        # Clean up
        df = df.drop(columns=["drug_name_extracted", "drug_name_match"], errors="ignore")
        
        matched = df["has_nadac_price"].sum()
        print(f"Matched {matched} drugs to NADAC pricing ({matched/len(df)*100:.1f}%)")
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional derived features."""
        
        # Review priority encoding
        df["is_priority_review"] = df["review_priority"].str.upper() == "PRIORITY"
        df["is_standard_review"] = df["review_priority"].str.upper() == "STANDARD"
        
        # Application type encoding
        df["is_nda"] = df["application_type"] == "NDA"
        df["is_bla"] = df["application_type"] == "BLA"
        df["is_anda"] = df["application_type"] == "ANDA"
        
        # Submission class features (new molecular entity, etc.)
        if "submission_class_code_description" in df.columns:
            df["is_new_molecular_entity"] = df["submission_class_code_description"].str.contains(
                "new molecular entity|nme", case=False, na=False
            )
            df["is_new_indication"] = df["submission_class_code_description"].str.contains(
                "new indication|efficacy", case=False, na=False
            )
        
        # Year features
        if "submission_status_date" in df.columns:
            df["approval_year"] = df["submission_status_date"].dt.year
            df["approval_month"] = df["submission_status_date"].dt.month
            df["approval_quarter"] = df["submission_status_date"].dt.quarter
        
        # Route of administration features
        if "route" in df.columns:
            df["is_oral"] = df["route"].str.contains("oral", case=False, na=False)
            df["is_injection"] = df["route"].str.contains(
                "injection|intravenous|subcutaneous|intramuscular", case=False, na=False
            )
            df["is_topical"] = df["route"].str.contains("topical|dermal", case=False, na=False)
        
        return df
    
    def run_preprocessing(self) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        print("="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load all data sources
        print("\n[1/5] Loading data sources...")
        fda_df = self.load_fda_approvals()
        orphan_df = self.load_orphan_drugs()
        cms_df = self.load_cms_coverage()
        nadac_df = self.load_nadac_pricing()
        
        # Merge data sources
        print("\n[2/5] Merging orphan drug status...")
        df = self.merge_orphan_status(fda_df, orphan_df)
        
        print("\n[3/5] Adding coverage features...")
        df = self.add_coverage_features(df, cms_df)
        
        print("\n[4/5] Adding pricing features...")
        df = self.add_pricing_features(df, nadac_df)
        
        print("\n[5/5] Creating derived features...")
        df = self.create_derived_features(df)
        
        # Save processed data
        output_path = self.processed_dir / "merged_drug_data.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
        
        # Save metadata
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "total_records": len(df),
            "columns": list(df.columns),
            "feature_counts": {
                "is_orphan": int(df["is_orphan"].sum()),
                "is_priority_review": int(df["is_priority_review"].sum()) if "is_priority_review" in df.columns else 0,
                "has_nadac_price": int(df["has_nadac_price"].sum()) if "has_nadac_price" in df.columns else 0,
            }
        }
        
        metadata_path = self.processed_dir / "preprocessing_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return df


def main():
    """Main entry point."""
    preprocessor = DataPreprocessor()
    df = preprocessor.run_preprocessing()
    
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nKey feature distributions:")
    
    bool_cols = [col for col in df.columns if df[col].dtype == bool]
    for col in bool_cols[:10]:
        print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    main()
