"""
FDA Drug Approvals Data Collection

Fetches drug approval data from the openFDA API including:
- Approval dates
- Submission types (NDA, BLA, ANDA)
- Review classification (Priority, Standard)
- Therapeutic equivalence codes
- Sponsor information
"""

import requests
import pandas as pd
from datetime import datetime
import time
import json
from pathlib import Path
from tqdm import tqdm


class FDAApprovalsCollector:
    """Collect drug approval data from openFDA API."""
    
    BASE_URL = "https://api.fda.gov/drug/drugsfda.json"
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_approvals_by_year(self, year: int, limit: int = 100) -> list:
        """Fetch all approvals for a given year."""
        all_results = []
        skip = 0
        
        while True:
            params = {
                "search": f"submissions.submission_status_date:[{year}0101+TO+{year}1231]",
                "limit": limit,
                "skip": skip
            }
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                    
                all_results.extend(results)
                skip += limit
                
                # Check if we've gotten all results
                total = data.get("meta", {}).get("results", {}).get("total", 0)
                if skip >= total:
                    break
                    
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {year} (skip={skip}): {e}")
                break
        
        return all_results
    
    def parse_approval_record(self, record: dict) -> list:
        """Parse a single FDA record into flat rows (one per submission)."""
        rows = []
        
        application_number = record.get("application_number", "")
        sponsor_name = record.get("sponsor_name", "")
        
        # Get product info
        products = record.get("products", [])
        brand_name = products[0].get("brand_name", "") if products else ""
        generic_name = products[0].get("active_ingredients", [{}])[0].get("name", "") if products else ""
        dosage_form = products[0].get("dosage_form", "") if products else ""
        route = products[0].get("route", "") if products else ""
        
        # Get therapeutic equivalence if available
        te_code = ""
        if products and products[0].get("te_code"):
            te_code = products[0].get("te_code", "")
        
        # Process each submission
        submissions = record.get("submissions", [])
        for sub in submissions:
            # Only include original approvals and supplements, not amendments
            sub_type = sub.get("submission_type", "")
            sub_status = sub.get("submission_status", "")
            
            if sub_status not in ["AP", "TA"]:  # Approved or Tentative Approval
                continue
            
            row = {
                "application_number": application_number,
                "sponsor_name": sponsor_name,
                "brand_name": brand_name,
                "generic_name": generic_name,
                "dosage_form": dosage_form,
                "route": route,
                "te_code": te_code,
                "submission_type": sub_type,
                "submission_number": sub.get("submission_number", ""),
                "submission_status": sub_status,
                "submission_status_date": sub.get("submission_status_date", ""),
                "review_priority": sub.get("review_priority", ""),
                "submission_class_code": sub.get("submission_class_code", ""),
                "submission_class_code_description": sub.get("submission_class_code_description", ""),
            }
            
            # Extract application docs info if available
            app_docs = sub.get("application_docs", [])
            row["has_label"] = any(d.get("type") == "Label" for d in app_docs)
            row["has_letter"] = any(d.get("type") == "Letter" for d in app_docs)
            
            rows.append(row)
        
        return rows
    
    def collect_all_approvals(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Collect all approvals for a range of years."""
        all_rows = []
        
        print(f"Collecting FDA approvals from {start_year} to {end_year}...")
        
        for year in tqdm(range(start_year, end_year + 1), desc="Years"):
            results = self.fetch_approvals_by_year(year)
            print(f"  {year}: {len(results)} records")
            
            for record in results:
                rows = self.parse_approval_record(record)
                all_rows.extend(rows)
        
        df = pd.DataFrame(all_rows)
        
        # Convert date column
        if "submission_status_date" in df.columns:
            df["submission_status_date"] = pd.to_datetime(
                df["submission_status_date"], 
                format="%Y%m%d", 
                errors="coerce"
            )
        
        # Add derived columns
        if "application_number" in df.columns:
            df["application_type"] = df["application_number"].str.extract(r"^([A-Z]+)")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "fda_approvals.csv"):
        """Save the collected data."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
        
        # Also save summary stats
        summary = {
            "total_records": len(df),
            "date_range": {
                "min": str(df["submission_status_date"].min()) if "submission_status_date" in df.columns else None,
                "max": str(df["submission_status_date"].max()) if "submission_status_date" in df.columns else None,
            },
            "application_types": df["application_type"].value_counts().to_dict() if "application_type" in df.columns else {},
            "review_priorities": df["review_priority"].value_counts().to_dict() if "review_priority" in df.columns else {},
            "collected_at": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "fda_approvals_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to {summary_path}")
        
        return filepath


def main():
    """Main entry point."""
    collector = FDAApprovalsCollector(output_dir="data/raw")
    
    # Collect approvals from 2010-2024
    df = collector.collect_all_approvals(start_year=2010, end_year=2024)
    
    # Save results
    collector.save_data(df)
    
    # Print summary
    print("\n" + "="*50)
    print("COLLECTION SUMMARY")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"\nApplication types:")
    print(df["application_type"].value_counts())
    print(f"\nReview priorities:")
    print(df["review_priority"].value_counts())
    print(f"\nSubmission types:")
    print(df["submission_type"].value_counts().head(10))
    
    return df


if __name__ == "__main__":
    main()
