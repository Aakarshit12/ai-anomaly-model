import joblib
import os
import sys
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, '..')
models_dir = os.path.join(root_dir, 'models')

def export_isolation_forest(model_path, output_path):
    print("Exporting IsolationForest to ONNX...")
    model = joblib.load(model_path)
    initial_type = [('float_input', FloatTensorType([None, 10]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 17, 'ai.onnx.ml': 3})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    size = os.path.getsize(output_path)
    print(f"Saved to {output_path} ({size / 1024:.1f} KB)")

def export_sgd_classifier(model_path, output_path):
    print("Exporting SGDClassifier to ONNX...")
    model = joblib.load(model_path)
    initial_type = [('float_input', FloatTensorType([None, 10]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 17, 'ai.onnx.ml': 3})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    size = os.path.getsize(output_path)
    print(f"Saved to {output_path} ({size / 1024:.1f} KB)")

if __name__ == '__main__':
    export_isolation_forest(
        os.path.join(models_dir, 'isolation_forest.pkl'),
        os.path.join(models_dir, 'isolation_forest.onnx')
    )
    export_sgd_classifier(
        os.path.join(models_dir, 'sgd_classifier.pkl'),
        os.path.join(models_dir, 'sgd_classifier.onnx')
    )
    print("\nExport complete.")