# ai-anomaly-model

Training pipeline for two ML models used in the ci-cd-ai gateway for real-time network intrusion detection. Models are trained on NSL-KDD, exported to ONNX, and published as GitHub Release assets.

![Train and Publish](https://github.com/Aakarshit12/ai-anomaly-model/actions/workflows/train_and_publish.yml/badge.svg)

---

## What this repo does

Trains IsolationForest and SGDClassifier on the NSL-KDD network intrusion dataset and publishes frozen ONNX files as GitHub Release assets. The ci-cd-ai gateway downloads these files at Docker build time using wget.

---

## Dataset — NSL-KDD

- **Source:** University of New Brunswick — https://www.unb.ca/cic/datasets/nsl.html
- **Size:** 125,973 training / 22,544 test samples
- **Features:** 41 network traffic features + label
- **Labels:** normal, DoS, Probe, R2L, U2R → mapped to binary (0=normal, 1=attack)

### Manual Setup
```bash
# Place these files in data/
data/KDDTrain+.txt
data/KDDTest+.txt
```

See `data/README.md` for full download instructions.

---

## Feature Mapping

| Gateway Feature | NSL-KDD Column | Reason |
|---|---|---|
| ip_hash | src_bytes % 100000 | Proxy for IP behavior |
| endpoint_hash | dst_bytes % 10000 | Proxy for destination pattern |
| http_method | protocol_type (encoded) | tcp=0, udp=1, icmp=2 |
| payload_size | src_bytes | Raw payload size |
| hour_of_day | duration % 24 | Proxy for timing |
| is_weekend | 0 (constant) | No time info in NSL-KDD |
| query_param_count | num_failed_logins | Proxy for suspicious queries |
| header_count | count | Connection count proxy |
| user_agent_hash | service encoded % 10000 | Proxy for client type |
| content_type_hash | flag encoded % 1000 | Proxy for content type |

---

## Model Card — IsolationForest

- **Type:** Unsupervised anomaly detection
- **Purpose:** Cold-start classification before live data is available
- **Hyperparameters:** contamination=0.1, n_estimators=100, random_state=42
- **Precision:** ~80%
- **Recall:** ~24% (expected for unsupervised — no attack labels used)
- **Inference P99:** ~5ms

---

## Model Card — SGDClassifier

- **Type:** Supervised binary classification
- **Purpose:** Attack detection with online learning via partial_fit
- **Hyperparameters:** loss=modified_huber, alpha=0.00001, eta0=0.1, learning_rate=adaptive
- **Precision:** ~89%
- **Recall:** ~80%
- **Inference P99:** ~0.4ms
- **Online learning:** Gateway retrains every 1000 live requests using partial_fit

---

## How to Run
```bash
# Full pipeline
make all

# Individual steps
make preprocess
make train
make evaluate
make export
make validate

# Run tests
make test

# Clean model files
make clean
```

---

## How ci-cd-ai Uses These Models

At Docker build time the gateway downloads the latest ONNX files:
```bash
wget https://github.com/Aakarshit12/ai-anomaly-model/releases/latest/download/isolation_forest.onnx
wget https://github.com/Aakarshit12/ai-anomaly-model/releases/latest/download/sgd_classifier.onnx
```

Ensemble scoring:
```python
weight = min(live_request_count / 10000, 0.8)
final_score = (1 - weight) * iso_score + weight * sgd_score

# score > 0.7  → BLOCKED
# score 0.3-0.7 → SUSPICIOUS
# score < 0.3  → NORMAL
```

---

## GitHub Actions

| Workflow | Trigger | Purpose |
|---|---|---|
| train_and_publish.yml | push to main | Full pipeline + publish release |
| pr_checks.yml | pull request | Run tests only |