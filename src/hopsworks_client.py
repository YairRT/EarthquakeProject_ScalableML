"""
Hopsworks integration module for Feature Store and Model Registry.

This module provides clean, structured functions for:
- Connecting to Hopsworks
- Saving features to Feature Store
- Loading features from Feature Store
- Saving models to Model Registry
- Loading models from Model Registry
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root (works even if script is run from different directory)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Fix for hopsworks/hsfs compatibility issue
# hopsworks tries to import hsfs.hopsworks_udf which doesn't exist in newer versions
# We need to patch this BEFORE importing hopsworks
try:
    import hsfs
    if not hasattr(hsfs, 'hopsworks_udf'):
        # Create a dummy module to satisfy the import
        class DummyUDF:
            pass
        class DummyModule:
            udf = DummyUDF()
        hsfs.hopsworks_udf = DummyModule()
except (ImportError, AttributeError):
    pass  # If hsfs isn't available, continue anyway

# Constants
FEATURE_GROUP_NAME = "earthquake_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aftershock_logreg"


def get_feature_store():
    """
    Get Hopsworks Feature Store connection.
    
    Returns:
        FeatureStore: Hopsworks feature store instance
        
    Raises:
        ImportError: If hopsworks is not installed
        ValueError: If API key is not set
    """
    # Use hopsworks.login() instead of hsfs.connection() for better compatibility
    try:
        import hopsworks
    except ImportError:
        raise ImportError("hopsworks is not installed. Install with: pip install hopsworks")
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found. "
            "Set it in .env file or as environment variable."
        )
    
    # Use hopsworks.login() which is more reliable than hsfs.connection()
    project = hopsworks.login(api_key_value=api_key)
    return project.get_feature_store()


def get_model_registry():
    """
    Get Hopsworks Model Registry connection.
    
    Returns:
        ModelRegistry: Hopsworks model registry instance
        
    Raises:
        ImportError: If hopsworks is not installed
        ValueError: If API key is not set
    """
    # The hopsworks_udf fix is already applied at module level
    # so we can import hopsworks directly
    try:
        import hopsworks
    except ImportError:
        raise ImportError("hopsworks is not installed. Install with: pip install hopsworks")
    except AttributeError as e:
        # If there's still an AttributeError, the fix didn't work
        # This shouldn't happen since we fix it at module level
        if 'hopsworks_udf' in str(e):
            raise ImportError(
                "Hopsworks compatibility issue. "
                "The hopsworks_udf fix failed. "
                "Try updating hopsworks: pip install --upgrade hopsworks"
            ) from e
        raise
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found. "
            "Set it in .env file or as environment variable."
        )
    
    project = hopsworks.login(api_key_value=api_key)
    return project.get_model_registry()


def save_features_to_hopsworks(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Save features to Hopsworks Feature Store.
    
    Args:
        df: DataFrame with features to save. Must include 'event_id' and 'time' columns.
        
    Returns:
        tuple[bool, str]: (success, error_message)
        - success: True if successful, False otherwise
        - error_message: Error message if failed, empty string if successful
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    try:
        # Check required columns
        if 'event_id' not in df.columns:
            return False, "DataFrame missing 'event_id' column"
        if 'time' not in df.columns:
            return False, "DataFrame missing 'time' column"
        
        fs = get_feature_store()
        
        # Get or create feature group
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["event_id"],
            event_time="time",
            description="Earthquake events with engineered features and aftershock labels",
            online_enabled=True,  # Enable online store for faster reads
        )
        
        # Remove duplicates and insert
        to_write = df.drop_duplicates(subset=["event_id"]).copy()
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(to_write['time']):
            to_write['time'] = pd.to_datetime(to_write['time'], utc=True, errors='coerce')
        
        # Remove rows with invalid time
        to_write = to_write.dropna(subset=['time'])
        
        if to_write.empty:
            return False, "No valid data to write after cleaning (all rows had invalid time)"
        
        fg.insert(to_write, write_options={"wait_for_job": False})
        
        return True, ""
    except Exception as e:
        error_msg = str(e)
        # Return a more user-friendly error message
        if "HOPSWORKS_API_KEY" in error_msg:
            return False, "HOPSWORKS_API_KEY not found. Check your .env file."
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return False, f"Connection error: {error_msg}"
        else:
            return False, f"Error: {error_msg}"


def load_features_from_hopsworks(
    filter_start_time: Optional[pd.Timestamp] = None,
    filter_end_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Load features from Hopsworks Feature Store.
    
    Args:
        filter_start_time: Optional start time filter
        filter_end_time: Optional end time filter
        
    Returns:
        DataFrame: Loaded features, or empty DataFrame if not found/failed
    """
    try:
        fs = get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        # Build query
        query = fg.select_all()
        if filter_start_time:
            query = query.filter(fg.time >= filter_start_time)
        if filter_end_time:
            query = query.filter(fg.time <= filter_end_time)
        
        # Try to read (will fail gracefully if feature group is empty)
        df = query.read()
        
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
        
        return df
        
    except Exception:
        # Feature group doesn't exist or reading failed
        return pd.DataFrame()


def save_model_to_hopsworks(model, metrics: dict) -> bool:
    """
    Save trained model to Hopsworks Model Registry.
    
    Args:
        model: Trained model (sklearn Pipeline)
        metrics: Dictionary with model metrics (e.g., {'auc': 0.85, 'n_samples': 1000})
        
    Returns:
        bool: True if successful, False otherwise
    """
    import joblib
    import tempfile
    import shutil
    
    try:
        mr = get_model_registry()
        
        # Create temporary directory for model files
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            joblib.dump(model, model_path)
            
            # Create model in registry
            model_reg = mr.python.create_model(
                name=MODEL_NAME,
                metrics=metrics,
                description="Logistic regression model for aftershock prediction",
            )
            
            # Upload model
            model_reg.save(model_path)
        
        return True
    except Exception as e:
        print(f"Error saving model to Hopsworks: {e}")
        return False


def load_model_from_hopsworks():
    """
    Load the latest model from Hopsworks Model Registry.
    
    Returns:
        model: Trained model (sklearn Pipeline)
        
    Raises:
        ValueError: If no model found in registry
    """
    import joblib
    
    mr = get_model_registry()
    models = mr.get_models(name=MODEL_NAME)
    
    if not models:
        raise ValueError(f"No models found in registry for name '{MODEL_NAME}'")
    
    # Get latest version
    latest = max(models, key=lambda m: m.version)
    model_dir = latest.download()
    
    # Load model
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def get_model_info():
    """
    Get model information including metrics from Hopsworks Model Registry.
    
    Returns:
        dict: Model information including metrics, version, etc.
        
    Raises:
        ValueError: If no model found in registry
    """
    mr = get_model_registry()
    models = mr.get_models(name=MODEL_NAME)
    
    if not models:
        raise ValueError(f"No models found in registry for name '{MODEL_NAME}'")
    
    # Get latest version
    latest = max(models, key=lambda m: m.version)
    
    # Get metrics
    metrics = latest.training_metrics if hasattr(latest, 'training_metrics') else {}
    
    return {
        'version': latest.version,
        'metrics': metrics,
        'created': latest.created if hasattr(latest, 'created') else None,
        'description': latest.description if hasattr(latest, 'description') else None,
    }

