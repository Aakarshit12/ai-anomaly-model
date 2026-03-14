import numpy as np
import onnxruntime as rt
import time
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, '..')
models_dir = os.path.join(root_dir, 'models')

def validate_model(onnx_path, model_name):
    print(f"\nValidating {model_name}...")
    sess = rt.InferenceSession(onnx_path)

    for inp in sess.get_inputs():
        print(f"  Input  : {inp.name} | shape: {inp.shape} | type: {inp.type}")
    for out in sess.get_outputs():
        print(f"  Output : {out.name} | shape: {out.shape} | type: {out.type}")

    input_name = sess.get_inputs()[0].name

    # Warmup phase — lets ONNX runtime JIT compile before measuring
    print("  Warming up...")
    for _ in range(20):
        sample = np.random.rand(1, 10).astype(np.float32)
        sess.run(None, {input_name: sample})

    # Now measure 100 inferences
    latencies = []
    for _ in range(100):
        sample = np.random.rand(1, 10).astype(np.float32)
        start = time.perf_counter()
        sess.run(None, {input_name: sample})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    print(f"  Min    : {latencies.min():.3f} ms")
    print(f"  Mean   : {latencies.mean():.3f} ms")
    print(f"  P95    : {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99    : {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max    : {latencies.max():.3f} ms")

    p99 = np.percentile(latencies, 99)
    assert p99 < 10.0, f"FAIL: P99 latency {p99:.3f}ms exceeds 10ms limit!"
    print(f"  PASSED: P99 {p99:.3f}ms is under 10ms")
if __name__ == '__main__':
    validate_model(
        os.path.join(models_dir, 'isolation_forest.onnx'),
        'IsolationForest'
    )
    validate_model(
        os.path.join(models_dir, 'sgd_classifier.onnx'),
        'SGDClassifier'
    )
    print("\nAll models validated successfully.")