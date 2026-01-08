"""
CMS Medicare Coverage Data Collection

Fetches National Coverage Determinations (NCDs) from the CMS Coverage API.
NCDs are coverage policies that apply nationally to all Medicare beneficiaries.

API Documentation: https://api.coverage.cms.gov/
"""

import requests
import pandas as pd
from datetime import datetime
import time
import json
from pathlib import Path
from tqdm import tqdm


class CMSCoverageCollector:
    """Collect coverage data from CMS Coverage API."""
    
    # CMS Coverage API endpoints
    BASE_URL = "https://api.coverage.cms.gov"
    NCD_ENDPOINT = "/v1/ncd"
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_all_ncds(self, limit: int = 100) -> list:
        """Fetch all National Coverage Determinations."""
        all_results = []
        offset = 0
        
        print("Fetching National Coverage Determinations from CMS...")
        
        while True:
            url = f"{self.BASE_URL}{self.NCD_ENDPOINT}"
            params = {
                "limit": limit,
                "offset": offset
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # The API returns data in a 'data' field
                results = data.get("data", [])
                if not results:
                    break
                
                all_results.extend(results)
                print(f"  Fetched {len(all_results)} NCDs so far...")
                
                offset += limit
                
                # Check if we've gotten all results
                total = data.get("meta", {}).get("total", 0)
                if offset >= total or len(results) < limit:
                    break
                
                time.sleep(0.3)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching NCDs (offset={offset}): {e}")
                # Try to continue from where we left off
                if offset > 0:
                    break
                raise
        
        return all_results
    
    def fetch_ncd_details(self, ncd_id: str) -> dict:
        """Fetch detailed information for a specific NCD."""
        url = f"{self.BASE_URL}{self.NCD_ENDPOINT}/{ncd_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NCD {ncd_id}: {e}")
            return {}
    
    def parse_ncd_record(self, record: dict) -> dict:
        """Parse an NCD record into a flat structure."""
        return {
            "ncd_id": record.get("ncdId", ""),
            "ncd_version": record.get("ncdVersion", ""),
            "title": record.get("title", ""),
            "category": record.get("category", ""),
            "subcategory": record.get("subcategory", ""),
            "benefit_category": record.get("benefitCategory", ""),
            "ncd_status": record.get("ncdStatus", ""),
            "effective_date": record.get("effectiveDate", ""),
            "ending_effective_date": record.get("endingEffectiveDate", ""),
            "implementation_date": record.get("implementationDate", ""),
            "last_modified_date": record.get("lastModifiedDate", ""),
            "manual_section_number": record.get("manualSectionNumber", ""),
            "manual_section_title": record.get("manualSectionTitle", ""),
            "is_current": record.get("isCurrent", False),
        }
    
    def collect_all_ncds(self) -> pd.DataFrame:
        """Collect all NCDs and return as DataFrame."""
        raw_ncds = self.fetch_all_ncds()
        
        print(f"\nParsing {len(raw_ncds)} NCD records...")
        
        rows = []
        for ncd in tqdm(raw_ncds, desc="Parsing NCDs"):
            row = self.parse_ncd_record(ncd)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert date columns
        date_columns = ["effective_date", "ending_effective_date", 
                        "implementation_date", "last_modified_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "cms_ncds.csv"):
        """Save the collected data."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} NCDs to {filepath}")
        
        # Save summary
        summary = {
            "total_records": len(df),
            "current_ncds": int(df["is_current"].sum()) if "is_current" in df.columns else 0,
            "categories": df["category"].value_counts().to_dict() if "category" in df.columns else {},
            "statuses": df["ncd_status"].value_counts().to_dict() if "ncd_status" in df.columns else {},
            "date_range": {
                "min_effective": str(df["effective_date"].min()) if "effective_date" in df.columns else None,
                "max_effective": str(df["effective_date"].max()) if "effective_date" in df.columns else None,
            },
            "collected_at": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "cms_ncds_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to {summary_path}")
        
        return filepath


def main():
    """Main entry point."""
    collector = CMSCoverageCollector(output_dir="data/raw")
    
    # Collect all NCDs
    df = collector.collect_all_ncds()
    
    # Save results
    collector.save_data(df)
    
    # Print summary
    print("\n" + "="*50)
    print("CMS NCD COLLECTION SUMMARY")
    print("="*50)
    print(f"Total NCDs: {len(df)}")
    print(f"Current NCDs: {df['is_current'].sum()}")
    print(f"\nCategories:")
    print(df["category"].value_counts())
    print(f"\nStatuses:")
    print(df["ncd_status"].value_counts())
    
    return df


if __name__ == "__main__":
    main()
