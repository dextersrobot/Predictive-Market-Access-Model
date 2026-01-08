"""
Market Access Predictor - Data Collection Pipeline

This script orchestrates the collection of all data sources needed
for the predictive market access model.

Data Sources:
1. FDA Drug Approvals (openFDA API)
2. CMS National Coverage Determinations (CMS Coverage API)
3. FDA Orphan Drug Designations (manual download required)
4. NADAC Drug Pricing (Medicaid.gov API)

Usage:
    python collect_all_data.py           # Collect all available data
    python collect_all_data.py --quick   # Quick mode (recent data only)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.fda_approvals import FDAApprovalsCollector
from data_collection.cms_coverage import CMSCoverageCollector
from data_collection.orphan_drugs import OrphanDrugCollector
from data_collection.nadac_pricing import NADACCollector


def collect_all_data(quick_mode: bool = False, output_dir: str = "data/raw"):
    """
    Collect all data sources for the market access predictor.
    
    Args:
        quick_mode: If True, collect smaller/recent datasets for faster iteration
        output_dir: Directory to save collected data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "collection_started": datetime.now().isoformat(),
        "quick_mode": quick_mode,
        "sources": {}
    }
    
    print("="*60)
    print("MARKET ACCESS PREDICTOR - DATA COLLECTION")
    print("="*60)
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"Output: {output_path.absolute()}")
    print("="*60)
    
    # 1. FDA Approvals
    print("\n[1/4] Collecting FDA Approvals...")
    print("-"*40)
    try:
        fda_collector = FDAApprovalsCollector(output_dir=output_dir)
        start_year = 2020 if quick_mode else 2010
        fda_df = fda_collector.collect_all_approvals(
            start_year=start_year, 
            end_year=2024
        )
        fda_collector.save_data(fda_df)
        results["sources"]["fda_approvals"] = {
            "status": "success",
            "records": len(fda_df),
            "date_range": f"{start_year}-2024"
        }
    except Exception as e:
        print(f"ERROR collecting FDA approvals: {e}")
        results["sources"]["fda_approvals"] = {"status": "error", "error": str(e)}
    
    # 2. CMS Coverage
    print("\n[2/4] Collecting CMS National Coverage Determinations...")
    print("-"*40)
    try:
        cms_collector = CMSCoverageCollector(output_dir=output_dir)
        cms_df = cms_collector.collect_all_ncds()
        cms_collector.save_data(cms_df)
        results["sources"]["cms_coverage"] = {
            "status": "success",
            "records": len(cms_df)
        }
    except Exception as e:
        print(f"ERROR collecting CMS coverage: {e}")
        results["sources"]["cms_coverage"] = {"status": "error", "error": str(e)}
    
    # 3. Orphan Drugs (template only - requires manual download)
    print("\n[3/4] Setting up Orphan Drug data structure...")
    print("-"*40)
    try:
        orphan_collector = OrphanDrugCollector(output_dir=output_dir)
        orphan_df = orphan_collector.create_manual_dataset()
        orphan_collector.save_data(orphan_df)
        results["sources"]["orphan_drugs"] = {
            "status": "template_created",
            "note": "Manual download required from FDA website"
        }
    except Exception as e:
        print(f"ERROR setting up orphan drugs: {e}")
        results["sources"]["orphan_drugs"] = {"status": "error", "error": str(e)}
    
    # 4. NADAC Pricing
    print("\n[4/4] Collecting NADAC Pricing data...")
    print("-"*40)
    try:
        nadac_collector = NADACCollector(output_dir=output_dir)
        days = 30 if quick_mode else 90
        raw_data = nadac_collector.fetch_recent_nadac(days=days)
        nadac_df = nadac_collector.process_nadac_data(raw_data)
        nadac_collector.save_data(nadac_df)
        results["sources"]["nadac_pricing"] = {
            "status": "success",
            "records": len(nadac_df),
            "days": days
        }
    except Exception as e:
        print(f"ERROR collecting NADAC pricing: {e}")
        results["sources"]["nadac_pricing"] = {"status": "error", "error": str(e)}
    
    # Save collection results
    results["collection_completed"] = datetime.now().isoformat()
    results_path = output_path / "collection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    
    for source, info in results["sources"].items():
        status = info.get("status", "unknown")
        if status == "success":
            print(f"✓ {source}: {info.get('records', 'N/A')} records")
        elif status == "template_created":
            print(f"⚠ {source}: Template created (manual download needed)")
        else:
            print(f"✗ {source}: {info.get('error', 'Failed')}")
    
    print(f"\nResults saved to: {results_path}")
    print("\nNext steps:")
    print("1. Download orphan drug data from FDA website")
    print("2. Run preprocessing pipeline: python src/preprocessing/merge_data.py")
    print("3. Run feature engineering: python src/features/engineer_features.py")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect all data for Market Access Predictor"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: collect recent/smaller datasets for faster iteration"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for collected data"
    )
    args = parser.parse_args()
    
    results = collect_all_data(
        quick_mode=args.quick,
        output_dir=args.output
    )
    
    # Exit with error code if any collection failed
    failed = sum(1 for s in results["sources"].values() if s.get("status") == "error")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
