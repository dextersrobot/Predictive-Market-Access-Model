"""
NADAC (National Average Drug Acquisition Cost) Data Collection

NADAC represents actual pharmacy acquisition costs, collected via CMS surveys.
This is one of the best publicly available drug pricing benchmarks.

Data source: https://data.medicaid.gov/nadac
Updated weekly by CMS.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
from io import StringIO
from tqdm import tqdm


class NADACCollector:
    """Collect NADAC drug pricing data from Medicaid.gov."""
    
    # Medicaid.gov NADAC API endpoint
    # Using the Socrata Open Data API
    NADAC_API = "https://data.medicaid.gov/resource/99315a95-37ac-4eee-946a-3c523b4c481e.json"
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_nadac_data(self, limit: int = 50000, offset: int = 0) -> list:
        """
        Fetch NADAC data from Medicaid.gov API.
        
        The API has a default limit, so we paginate through results.
        """
        all_results = []
        
        print("Fetching NADAC pricing data from Medicaid.gov...")
        
        while True:
            params = {
                "$limit": limit,
                "$offset": offset,
                "$order": "as_of_date DESC"
            }
            
            try:
                response = requests.get(self.NADAC_API, params=params, timeout=60)
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    break
                
                all_results.extend(results)
                print(f"  Fetched {len(all_results)} NADAC records...")
                
                if len(results) < limit:
                    break
                
                offset += limit
                
                # For demo purposes, limit total records
                # Remove this for full collection
                if len(all_results) >= 200000:
                    print("  Reached demo limit of 200,000 records")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching NADAC data: {e}")
                break
        
        return all_results
    
    def fetch_recent_nadac(self, days: int = 90) -> list:
        """Fetch only recent NADAC data (last N days)."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        all_results = []
        offset = 0
        limit = 50000
        
        print(f"Fetching NADAC data from last {days} days...")
        
        while True:
            params = {
                "$limit": limit,
                "$offset": offset,
                "$where": f"as_of_date >= '{cutoff_date}'",
                "$order": "as_of_date DESC"
            }
            
            try:
                response = requests.get(self.NADAC_API, params=params, timeout=60)
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    break
                
                all_results.extend(results)
                print(f"  Fetched {len(all_results)} recent NADAC records...")
                
                if len(results) < limit:
                    break
                
                offset += limit
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching NADAC data: {e}")
                break
        
        return all_results
    
    def parse_nadac_record(self, record: dict) -> dict:
        """Parse a NADAC record into standardized format."""
        return {
            "ndc": record.get("ndc", ""),
            "ndc_description": record.get("ndc_description", ""),
            "nadac_per_unit": self._safe_float(record.get("nadac_per_unit")),
            "effective_date": record.get("effective_date", ""),
            "pricing_unit": record.get("pricing_unit", ""),
            "pharmacy_type_indicator": record.get("pharmacy_type_indicator", ""),
            "otc": record.get("otc", ""),
            "explanation_code": record.get("explanation_code", ""),
            "classification_for_rate_setting": record.get("classification_for_rate_setting", ""),
            "as_of_date": record.get("as_of_date", ""),
        }
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def process_nadac_data(self, raw_data: list) -> pd.DataFrame:
        """Process raw NADAC data into DataFrame."""
        print(f"\nProcessing {len(raw_data)} NADAC records...")
        
        rows = [self.parse_nadac_record(r) for r in tqdm(raw_data, desc="Parsing")]
        df = pd.DataFrame(rows)
        
        # Convert date columns
        for col in ["effective_date", "as_of_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Extract NDC components
        if "ndc" in df.columns:
            # NDC format: XXXXX-XXXX-XX (labeler-product-package)
            df["ndc_labeler"] = df["ndc"].str.split("-").str[0]
            df["ndc_product"] = df["ndc"].str.split("-").str[1]
            df["ndc_package"] = df["ndc"].str.split("-").str[2]
        
        # Add classification
        if "classification_for_rate_setting" in df.columns:
            df["is_brand"] = df["classification_for_rate_setting"].str.upper() == "B"
            df["is_generic"] = df["classification_for_rate_setting"].str.upper() == "G"
        
        return df
    
    def create_price_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by drug."""
        if "ndc_description" not in df.columns or "nadac_per_unit" not in df.columns:
            return pd.DataFrame()
        
        summary = df.groupby("ndc_description").agg({
            "nadac_per_unit": ["mean", "min", "max", "std", "count"],
            "ndc": "first",
            "is_brand": "first",
            "as_of_date": "max"
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            "drug_description", "avg_price", "min_price", "max_price", 
            "price_std", "price_count", "ndc", "is_brand", "latest_date"
        ]
        
        return summary.sort_values("avg_price", ascending=False)
    
    def save_data(self, df: pd.DataFrame, filename: str = "nadac_prices.csv"):
        """Save the collected data."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
        
        # Save summary
        summary = {
            "total_records": len(df),
            "unique_ndcs": df["ndc"].nunique() if "ndc" in df.columns else 0,
            "date_range": {
                "min": str(df["as_of_date"].min()) if "as_of_date" in df.columns else None,
                "max": str(df["as_of_date"].max()) if "as_of_date" in df.columns else None,
            },
            "price_stats": {
                "mean": float(df["nadac_per_unit"].mean()) if "nadac_per_unit" in df.columns else None,
                "median": float(df["nadac_per_unit"].median()) if "nadac_per_unit" in df.columns else None,
                "max": float(df["nadac_per_unit"].max()) if "nadac_per_unit" in df.columns else None,
            },
            "classifications": df["classification_for_rate_setting"].value_counts().to_dict() if "classification_for_rate_setting" in df.columns else {},
            "collected_at": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "nadac_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to {summary_path}")
        
        return filepath


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect NADAC pricing data")
    parser.add_argument(
        "--recent-only",
        type=int,
        default=90,
        help="Only fetch data from last N days (default: 90)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fetch full historical data (warning: very large)"
    )
    args = parser.parse_args()
    
    collector = NADACCollector(output_dir="data/raw")
    
    if args.full:
        raw_data = collector.fetch_nadac_data()
    else:
        raw_data = collector.fetch_recent_nadac(days=args.recent_only)
    
    # Process data
    df = collector.process_nadac_data(raw_data)
    
    # Save full data
    collector.save_data(df, "nadac_prices.csv")
    
    # Create and save price summary
    price_summary = collector.create_price_summary(df)
    if not price_summary.empty:
        collector.save_data(price_summary, "nadac_price_summary.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("NADAC PRICING DATA SUMMARY")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Unique NDCs: {df['ndc'].nunique()}")
    print(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
    print(f"\nPrice statistics:")
    print(f"  Mean: ${df['nadac_per_unit'].mean():.4f}")
    print(f"  Median: ${df['nadac_per_unit'].median():.4f}")
    print(f"  Max: ${df['nadac_per_unit'].max():.2f}")
    print(f"\nClassifications:")
    print(df["classification_for_rate_setting"].value_counts())
    
    return df


if __name__ == "__main__":
    main()
