"""
Training script for aftershock prediction model.

This script:
1. Loads data from Hopsworks Feature Store
2. Trains a logistic regression model
3. Saves the model to Hopsworks Model Registry

Usage:
    python scripts/train_model.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hopsworks_client import (
    get_feature_store,
    save_model_to_hopsworks,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from src.models import train_logreg
import pandas as pd
from datetime import datetime, timedelta, timezone


def main():
    """Main training function."""
    print("=" * 60)
    print("Training Aftershock Prediction Model")
    print("=" * 60)
    
    # Step 1: Load data from Feature Store
    print("\n[1/3] Loading data from Hopsworks Feature Store...")
    try:
        fs = get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        # Read all data from feature store
        # Try with Hive if Flight doesn't work
        try:
            df = fg.read()
        except Exception:
            print("  ⚠️  Flight read failed, trying Hive...")
            df = fg.read(read_options={"use_hive": True})
        
        if df.empty:
            raise ValueError("Feature Store is empty. Run the Streamlit app first to collect data.")
        
        print(f"  ✓ Loaded {len(df)} rows from Feature Store")
        
    except Exception as e:
        print(f"  ✗ Error loading from Feature Store: {e}")
        print("\n  💡 Make sure you have:")
        print("     - Run the Streamlit app to collect data")
        print("     - Data has been saved to Feature Store")
        return
    
    # Step 2: Train model
    print("\n[2/3] Training model...")
    try:
        pipe, metrics, df_pred = train_logreg(df)
        print(f"  ✓ Model trained successfully!")
        print(f"\n  Model Metrics:")
        print(f"    - Samples: {metrics['n_samples']}")
        print(f"    - Positive labels: {metrics['n_pos']} ({metrics['pos_rate']:.1%})")
        print(f"    - AUC: {metrics['auc']:.3f}")
        print(f"\n  Classification Report:")
        print(metrics['report'])
        
    except ValueError as e:
        print(f"  ✗ Training failed: {e}")
        print("\n  💡 Try:")
        print("     - Increase time span in Streamlit app")
        print("     - Adjust T_hours or R_km parameters")
        print("     - Collect more data")
        return
    except Exception as e:
        print(f"  ✗ Training error: {e}")
        return
    
    # Step 3: Save model to Model Registry
    print("\n[3/3] Saving model to Hopsworks Model Registry...")
    try:
        # Prepare metrics (remove 'report' as it's a string, not a metric)
        model_metrics = {
            'auc': metrics['auc'],
            'n_samples': metrics['n_samples'],
            'n_pos': metrics['n_pos'],
            'pos_rate': metrics['pos_rate'],
        }
        
        success = save_model_to_hopsworks(pipe, model_metrics)
        
        if success:
            print(f"  ✓ Model saved to Model Registry!")
            print(f"\n  ✓ Training complete! Model is now available in the Streamlit app.")
        else:
            print(f"  ✗ Failed to save model to Model Registry")
            
    except Exception as e:
        print(f"  ✗ Error saving model: {e}")
        return


if __name__ == "__main__":
    main()

