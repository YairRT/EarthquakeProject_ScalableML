"""
Script to collect large amounts of earthquake data for model training.

This script automatically collects data from multiple time periods and regions
to build a comprehensive training dataset (target: 10,000+ samples).

Usage:
    python scripts/collect_training_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.usgs_client import get_earthquakes
from src.features import basic_time_feats, add_seq_feat
from src.labels import add_aftershock_label
from src.hopsworks_client import save_features_to_hopsworks
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# Regions to collect data from
REGIONS = {
    'Japan': (122, 24, 146, 46),
    'Mexico': (-118, 14, -86, 33),
    'Chile': (-76, -56, -66, -17),
    'California': (-125, 32, -113, 42),
    'Indonesia': (95, -11, 141, 6),
    'Global': None,
}

TARGET_SAMPLES = 10000
T_HOURS = 24
R_KM = 100


def collect_data_from_period(start_date, end_date, region_name, bbox, min_mag=3.0, limit=1000):
    """Collect data from a specific time period and region."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        print(f"  Fetching {region_name} ({start_str} to {end_str})...")
        df = get_earthquakes(
            starttime=start_str,
            endtime=end_str,
            min_magnitude=min_mag,
            bbox=bbox,
            limit=limit
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Process features
        df_feat = basic_time_feats(df)
        df_feat = add_seq_feat(df_feat)
        df_labeled = add_aftershock_label(df_feat, T_hours=T_HOURS, R_km=R_KM)
        df_labeled['region'] = region_name
        
        return df_labeled
    except Exception as e:
        print(f"  ⚠️  Error fetching {region_name}: {e}")
        return pd.DataFrame()


def main():
    """Main data collection function."""
    print("=" * 70)
    print("Collecting Training Data for Aftershock Prediction Model")
    print("=" * 70)
    print(f"\nTarget: {TARGET_SAMPLES:,} samples")
    print("This will collect data from multiple regions and time periods.\n")
    
    all_data = []
    total_collected = 0
    
    # Collect data from different time periods (going back in time)
    end_date = datetime.now(timezone.utc)
    
    # Collect from last 2 years, in 90-day chunks
    periods = []
    current_date = end_date
    for i in range(8):  # 8 periods of 90 days = ~2 years
        period_end = current_date
        period_start = current_date - timedelta(days=90)
        periods.append((period_start, period_end))
        current_date = period_start
    
    print(f"Collecting data from {len(periods)} time periods and {len(REGIONS)} regions...")
    print(f"Total queries: {len(periods) * len(REGIONS)}\n")
    
    query_count = 0
    for period_start, period_end in periods:
        for region_name, bbox in REGIONS.items():
            query_count += 1
            print(f"[{query_count}/{len(periods) * len(REGIONS)}] ", end="")
            
            df = collect_data_from_period(
                period_start, 
                period_end, 
                region_name, 
                bbox,
                min_mag=3.0,
                limit=1000
            )
            
            if not df.empty:
                all_data.append(df)
                total_collected += len(df)
                print(f"  ✓ Collected {len(df)} events (Total: {total_collected:,})")
                
                # Save to Hopsworks periodically
                if total_collected % 500 == 0:
                    print(f"  💾 Saving batch to Hopsworks...")
                    combined = pd.concat(all_data, ignore_index=True)
                    success, msg = save_features_to_hopsworks(combined)
                    if success:
                        print(f"  ✓ Saved to Hopsworks")
                    else:
                        print(f"  ⚠️  Save warning: {msg}")
                    all_data = []  # Clear to avoid duplicates
            else:
                print(f"  - No data for this period/region")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
            
            # Check if we have enough
            if total_collected >= TARGET_SAMPLES:
                print(f"\n✓ Target reached! Collected {total_collected:,} samples.")
                break
        
        if total_collected >= TARGET_SAMPLES:
            break
    
    # Combine all collected data
    if all_data:
        print(f"\n💾 Saving final batch to Hopsworks...")
        combined = pd.concat(all_data, ignore_index=True)
        success, msg = save_features_to_hopsworks(combined)
        if success:
            print(f"  ✓ Saved to Hopsworks")
        else:
            print(f"  ⚠️  Save warning: {msg}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Data Collection Complete!")
    print("=" * 70)
    print(f"Total samples collected: {total_collected:,}")
    
    if total_collected >= TARGET_SAMPLES:
        print(f"✓ Target of {TARGET_SAMPLES:,} samples reached!")
    else:
        print(f"⚠️  Only collected {total_collected:,} samples (target: {TARGET_SAMPLES:,})")
        print("   You may want to:")
        print("   - Increase time periods")
        print("   - Add more regions")
        print("   - Lower minimum magnitude threshold")
    
    print(f"\n💡 Next step: Train the model with:")
    print(f"   python scripts/train_model.py")


if __name__ == "__main__":
    main()

