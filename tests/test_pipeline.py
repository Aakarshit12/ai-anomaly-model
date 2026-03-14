import numpy as np
import pytest
import sys
import os

# Fix paths
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, '..')
src_dir  = os.path.join(root_dir, 'src')
sys.path.insert(0, src_dir)

from preprocess import extract_features, encode_categoricals
from export import export_isolation_forest, export_sgd_classifier
from validate_onnx import validate_model
import pandas as pd
import joblib
import tempfile

# --- Mock data helpers ---

def make_mock_df(n=200):
    np.random.seed(42)
    df = pd.DataFrame({
        'duration':          np.random.randint(0, 100, n),
        'protocol_type':     np.random.choice([0, 1, 2], n),
        'service':           np.random.randint(0, 50, n),
        'flag':              np.random.randint(0, 10, n),
        'src_bytes':         np.random.randint(0, 100000, n),
        'dst_bytes':         np.random.randint(0, 10000, n),
        'num_failed_logins': np.random.randint(0, 5, n),
        'count':             np.random.randint(0, 500, n),
    })
    return df

def make_mock_X_y(n=200):
    df = make_mock_df(n)
    X = extract_features(df).values
    y = np.random.randint(0, 2, n).astype(np.float32)
    return X, y

# --- Tests ---

def test_extract_features_shape():
    df = make_mock_df(100)
    X = extract_features(df)
    assert X.shape == (100, 10), f"Expected (100, 10), got {X.shape}"

def test_extract_features_dtype():
    df = make_mock_df(50)
    X = extract_features(df)
    assert X.dtypes.unique()[0] == np.float32

def test_train_isolation_forest():
    X, _ = make_mock_X_y(200)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'iso.pkl')
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=10)
        model.fit(X)
        joblib.dump(model, path)
        assert os.path.exists(path)
        loaded = joblib.load(path)
        preds = loaded.predict(X)
        assert set(preds).issubset({1, -1})

def test_train_sgd_classifier():
    X, y = make_mock_X_y(200)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'sgd.pkl')
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss='modified_huber', random_state=42, max_iter=100)
        model.fit(X, y)
        joblib.dump(model, path)
        assert os.path.exists(path)
        loaded = joblib.load(path)
        preds = loaded.predict(X)
        assert set(preds).issubset({0, 1, 0.0, 1.0})

def test_export_and_validate_isolation_forest():
    X, _ = make_mock_X_y(200)
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path  = os.path.join(tmpdir, 'iso.pkl')
        onnx_path = os.path.join(tmpdir, 'iso.onnx')
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=10)
        model.fit(X)
        joblib.dump(model, pkl_path)
        export_isolation_forest(pkl_path, onnx_path)
        assert os.path.exists(onnx_path)
        validate_model(onnx_path, 'IsolationForest-test')

def test_export_and_validate_sgd_classifier():
    X, y = make_mock_X_y(200)
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path  = os.path.join(tmpdir, 'sgd.pkl')
        onnx_path = os.path.join(tmpdir, 'sgd.onnx')
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss='modified_huber', random_state=42, max_iter=100)
        model.fit(X, y)
        joblib.dump(model, pkl_path)
        export_sgd_classifier(pkl_path, onnx_path)
        assert os.path.exists(onnx_path)
        validate_model(onnx_path, 'SGDClassifier-test')