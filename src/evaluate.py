import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import sys
import os

# Fix paths to work from any directory
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, '..')
models_dir = os.path.join(root_dir, 'models')
sys.path.insert(0, base_dir)

from preprocess import load_nslkdd

def evaluate_isolation_forest(model_path, X_test, y_test):
    print("\nEvaluating IsolationForest...")
    model = joblib.load(model_path)
    raw_preds = model.predict(X_test)
    # 1=normal, -1=anomaly → map to 0=normal, 1=attack
    preds = np.where(raw_preds == -1, 1, 0)

    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    cm        = confusion_matrix(y_test, preds)

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def evaluate_sgd_classifier(model_path, X_test, y_test):
    print("\nEvaluating SGDClassifier...")
    model = joblib.load(model_path)
    preds = model.predict(X_test)

    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    cm        = confusion_matrix(y_test, preds)

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {'precision': precision, 'recall': recall, 'f1': f1}
if __name__ == '__main__':
    train_path = os.path.join(root_dir, 'data', 'KDDTrain+.txt')
    test_path  = os.path.join(root_dir, 'data', 'KDDTest+.txt')

    X_train, X_test, y_train, y_test = load_nslkdd(train_path, test_path)

    iso_metrics = evaluate_isolation_forest(
        os.path.join(models_dir, 'isolation_forest.pkl'), X_test, y_test
    )
    sgd_metrics = evaluate_sgd_classifier(
        os.path.join(models_dir, 'sgd_classifier.pkl'), X_test, y_test
    )

    failed = False

    if sgd_metrics['precision'] < 0.85:
        print(f"\nFAIL: SGD precision {sgd_metrics['precision']:.4f} < 0.85")
        failed = True
    if sgd_metrics['recall'] < 0.80:
        print(f"\nFAIL: SGD recall {sgd_metrics['recall']:.4f} < 0.80")
        failed = True
    if iso_metrics['precision'] < 0.75:
        print(f"\nFAIL: IsolationForest precision {iso_metrics['precision']:.4f} < 0.75")
        failed = True

    if failed:
        print("\nEvaluation FAILED — models below threshold.")
        sys.exit(1)
    else:
        print("\nEvaluation PASSED — all models above threshold.")