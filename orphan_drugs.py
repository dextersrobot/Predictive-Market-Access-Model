"""
FDA Orphan Drug Designations Data Collection

The FDA Orphan Drug database doesn't have a public API, but we can scrape
the searchable database or use the bulk download approach.

This script fetches orphan drug designation data to identify rare disease drugs.
Orphan designation is a key predictor of market access outcomes due to:
- Limited competition
- 7-year market exclusivity
- Tax credits for development
- Reduced regulatory fees
"""

import requests
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from io import StringIO


class OrphanDrugCollector:
    """Collect orphan drug designation data from FDA."""
    
    # The FDA orphan drug database search endpoint
    SEARCH_URL = "https://www.accessdata.fda.gov/scripts/opdlisting/oopd/listResult.cfm"
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_manual_dataset(self) -> pd.DataFrame:
        """
        Create a reference dataset of orphan drug designations.
        
        Since the FDA orphan drug database requires browser interaction,
        this creates a structured dataset that can be enriched with
        manual downloads from the FDA website.
        
        To get full data:
        1. Go to https://www.accessdata.fda.gov/scripts/opdlisting/oopd/
        2. Search with no filters to get all records
        3. Export to Excel
        4. Save as orphan_drugs_manual.xlsx in data/raw/
        """
        
        # Create a template with expected columns
        columns = [
            "generic_name",
            "trade_name", 
            "designation_date",
            "designation_status",
            "orphan_designation",
            "fda_orphan_approval",
            "approved_indication",
            "marketing_status",
            "exclusivity_end_date",
            "sponsor_name",
            "sponsor_address",
            "contact_name"
        ]
        
        # Sample data structure (you'll replace with actual data)
        sample_data = {
            "generic_name": ["Example Drug A", "Example Drug B"],
            "trade_name": ["BrandA", "BrandB"],
            "designation_date": ["2020-01-15", "2021-06-30"],
            "designation_status": ["Designated", "Designated and Approved"],
            "orphan_designation": ["Treatment of rare cancer X", "Treatment of rare disease Y"],
            "fda_orphan_approval": ["Yes", "No"],
            "approved_indication": ["Rare cancer X", None],
            "marketing_status": ["Prescription", None],
            "exclusivity_end_date": ["2027-01-15", None],
            "sponsor_name": ["Pharma Co A", "Biotech B"],
            "sponsor_address": ["City, State", "City, State"],
            "contact_name": ["John Doe", "Jane Smith"]
        }
        
        df = pd.DataFrame(sample_data)
        
        print("\n" + "="*60)
        print("IMPORTANT: Manual Data Download Required")
        print("="*60)
        print("""
The FDA Orphan Drug database requires manual download:

1. Go to: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/
2. Click 'Search' with no filters to get all records
3. Click 'Export to Excel' button
4. Save the file as 'orphan_drugs_fda.xlsx' in the data/raw/ folder
5. Run this script again with --load-manual flag

For now, a template file has been created.
        """)
        
        return df
    
    def load_manual_download(self, filepath: str) -> pd.DataFrame:
        """Load manually downloaded orphan drug data."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Manual download not found at {path}. "
                "Please download from FDA website first."
            )
        
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
        
        # Parse dates
        date_columns = [col for col in df.columns if "date" in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df
    
    def create_orphan_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a lookup table for matching drugs to orphan status.
        
        This will be used to join with FDA approval data.
        """
        lookup = df.copy()
        
        # Create standardized drug name for matching
        if "generic_name" in lookup.columns:
            lookup["drug_name_std"] = (
                lookup["generic_name"]
                .str.lower()
                .str.strip()
                .str.replace(r"[^\w\s]", "", regex=True)
            )
        
        # Add binary flags
        if "designation_status" in lookup.columns:
            lookup["is_orphan_designated"] = True
            lookup["is_orphan_approved"] = lookup["designation_status"].str.contains(
                "approved", case=False, na=False
            )
        
        return lookup
    
    def save_data(self, df: pd.DataFrame, filename: str = "orphan_drugs.csv"):
        """Save the collected/processed data."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
        
        # Save summary
        summary = {
            "total_records": len(df),
            "columns": list(df.columns),
            "sample_generic_names": df["generic_name"].head(10).tolist() if "generic_name" in df.columns else [],
            "created_at": datetime.now().isoformat(),
            "note": "Template file - replace with FDA download"
        }
        
        summary_path = self.output_dir / "orphan_drugs_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")
        
        return filepath


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect FDA Orphan Drug data")
    parser.add_argument(
        "--load-manual", 
        type=str, 
        help="Path to manually downloaded FDA orphan drug file"
    )
    args = parser.parse_args()
    
    collector = OrphanDrugCollector(output_dir="data/raw")
    
    if args.load_manual:
        print(f"Loading manual download from {args.load_manual}...")
        df = collector.load_manual_download(args.load_manual)
        print(f"Loaded {len(df)} orphan drug records")
    else:
        print("Creating template dataset...")
        df = collector.create_manual_dataset()
    
    # Create lookup table
    lookup = collector.create_orphan_lookup(df)
    
    # Save both raw and lookup
    collector.save_data(df, "orphan_drugs_raw.csv")
    collector.save_data(lookup, "orphan_drugs_lookup.csv")
    
    # Print summary
    print("\n" + "="*50)
    print("ORPHAN DRUG DATA SUMMARY")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    main()
