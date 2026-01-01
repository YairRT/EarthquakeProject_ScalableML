import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn. model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['magnitude', 'depth', 'time_since_prev_hours',
                'distance_to_prev_km', 'rolling_count_6h', 'rolling_count_24h']



def train_logreg(df: pd.DataFrame, label_col = 'y_aftershock', test_size = 0.2, seed = 42):
    '''
    Here, we'll be training the logistic regression on the mentioned features

    returns pipeline, metrics_dict, df with pred
    '''

    data = df.copy()

    # Let's drop the columns that we are not using
    # We can just get the features from the feat store in Hopsworks and avoid this step

    keep_cols = [label_col] + FEATURE_COLS
    data = data.dropna(subset=keep_cols)

    X = data[FEATURE_COLS]
    y = data[label_col].astype(int)

    # Let's implement a basic filter to avoid training on a few positives (unstable training)
    pos = int(y.sum())

    if pos < 20:
        raise ValueError(f'Too few posiitive lables (y=1): {pos}. Increase time span or adjust R (T or R)')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,stratify=y)

    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))])

    pipe.fit(X_train, y_train)

    # Evaluate
    p_test = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, p_test)

    metrics = {
        'n_samples': int(len(data)),
        'n_pos':int(pos),
        'pos_rate':float(pos/len(data)),
        'auc': float(auc),
        'report': classification_report(y_test, (p_test >= 0.5).astype(int), output_dict=False)
    }


    data = data.copy()
    data['p_aftershock'] = pipe.predict_proba(X)[:,1]

    return pipe, metrics, data

def predict_aftershock_proba(model, df: pd.DataFrame):
    # Predict prob for each row and returns a Series

    X = df.reindex(columns=FEATURE_COLS)
    mask = ~X.isna().any(axis=1)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    if mask.any():
        out.loc[mask] = model.predict_proba(X.loc[mask])[:,1]
    return out


if __name__=='__main__':
    from usgs_client import get_earthquakes
    from datetime import datetime, timedelta
    from features import basic_time_feats, compute_freq_series, add_seq_feat
    from labels import add_aftershock_label

    days=30
    limit = 100

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')

    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=None,limit=limit)
    seq = basic_time_feats(df)
    seq = add_seq_feat(seq)
    processed = add_aftershock_label(seq)

    try:
        pipe, metrics, data_pred = train_logreg(processed)
        print("\nModel metrics:")
        for key, value in metrics.items():
            if key == 'report':
                print(f"{key}:\n{value}")
            else:
                print(f"{key}: {value}")
        print('\nData with predicted probabilities:')
        print(data_pred.head())
    except ValueError as e:
        print(f'Error: {e}')

