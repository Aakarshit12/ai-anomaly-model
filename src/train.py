import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.utils import compute_class_weight, shuffle
import joblib
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, '..')
models_dir = os.path.join(root_dir, 'models')
sys.path.insert(0, base_dir)

from preprocess import load_nslkdd

def train_isolation_forest(X_train):
    print("Training IsolationForest...")
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'isolation_forest.pkl')
    joblib.dump(model, path)
    print(f"IsolationForest saved to {path}")
    return model

def train_sgd_classifier(X_train, y_train):
    print("Training SGDClassifier...")

    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    sample_weights = np.array([class_weight_dict[int(c)] for c in y_train])

    model = SGDClassifier(
        loss='modified_huber',
        random_state=42,
        max_iter=1000,
        alpha=0.00001,
        eta0=0.1,
        learning_rate='adaptive',
        tol=1e-4
    )

    for epoch in range(5):
        X_s, y_s, sw_s = shuffle(X_train, y_train, sample_weights, random_state=epoch)
        model.partial_fit(X_s, y_s, classes=classes, sample_weight=sw_s)
        print(f"  Epoch {epoch+1}/5 done")

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'sgd_classifier.pkl')
    joblib.dump(model, path)
    print(f"SGDClassifier saved to {path}")
    return model

if __name__ == '__main__':
    train_path = os.path.join(root_dir, 'data', 'KDDTrain+.txt')
    test_path  = os.path.join(root_dir, 'data', 'KDDTest+.txt')

    X_train, X_test, y_train, y_test = load_nslkdd(train_path, test_path)
    train_isolation_forest(X_train)
    train_sgd_classifier(X_train, y_train)
    print("Training complete.")