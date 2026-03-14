# Architecture Decision Records

## 1. Why a separate repo for model training?

Keeping training in a separate repo (ai-anomaly-model) and inference in
ci-cd-ai follows the single responsibility principle. The training repo
owns the full ML pipeline — data, preprocessing, training, evaluation,
and export. The inference repo owns the gateway, API, and runtime.

This means:
- Model updates never require redeploying the gateway code
- Gateway deploys never retrigger expensive model retraining
- Each repo has its own CI/CD pipeline, dependencies, and release cycle
- The ONNX files are the clean contract between the two repos

## 2. Why NSL-KDD over synthetic data?

NSL-KDD is the standard benchmark for network intrusion detection research.
It has several advantages over synthetic data:

- Real network traffic patterns captured from actual attacks
- Balanced representation of attack types: DoS, Probe, R2L, U2R
- Cleaned version of KDD Cup 99 with duplicates removed
- Widely used in research so results are comparable
- Free and publicly available

Synthetic data would be faster to generate but would not capture the
complex feature correlations present in real network traffic. NSL-KDD
gives the models a realistic baseline to start from.

## 3. Why IsolationForest + SGDClassifier ensemble?

Each model solves a different problem:

IsolationForest is unsupervised — it learns what normal traffic looks
like without needing attack labels. This makes it ideal for cold start
when the gateway has no live traffic history yet. It catches novel
anomalies that fall outside the normal distribution.

SGDClassifier is supervised — it learns explicit attack patterns from
labeled NSL-KDD data. It is more precise than IsolationForest but
depends on having seen similar attack patterns during training.

The ensemble combines both:
- Early on: IsolationForest dominates (weight close to 0)
- Over time: SGDClassifier takes over as it trains on live traffic
- Together they cover both known and unknown attack patterns

## 4. Why ONNX over raw sklearn pickle at inference time?

Sklearn pickle files have several problems for production inference:

- Pickle files are tied to the exact Python and sklearn version used
  during training — version mismatches cause silent errors or crashes
- Pickle loading is slow and not optimized for inference
- Pickle files can execute arbitrary code on load — a security risk

ONNX solves all of these:
- ONNX Runtime is language and framework agnostic
- Inference is faster — ONNX Runtime applies graph optimizations
- No sklearn dependency needed in the gateway at all
- Cross-platform — same file runs on Linux, Windows, ARM
- Our validation confirmed P99 under 5ms for IsolationForest
  and under 0.4ms for SGDClassifier

## 5. Why this specific feature mapping from NSL-KDD's 41 features to 10?

The gateway extracts 10 features from live HTTP requests. NSL-KDD has
41 network packet level features. A direct mapping is impossible since
HTTP requests and raw network packets are different abstractions.

The mapping approximates the same information at a different level:

- src_bytes approximates payload size and IP behavior
- dst_bytes approximates destination patterns
- protocol_type maps cleanly to HTTP method encoding
- num_failed_logins maps to suspicious query parameter counts
- count maps to header count as both measure connection volume
- service and flag encode client and connection type information

The mapping is intentionally approximate. The goal is not perfect
feature alignment but a reasonable training signal that teaches the
models to distinguish normal from anomalous traffic patterns. The
gateway features are then normalized using the same StandardScaler
fitted during training.

## 6. Why partial_fit / online learning for the SGD model?

The NSL-KDD dataset is from 1999. Attack patterns change over time.
A model trained only on historical data will degrade as new attack
types emerge that it has never seen.

SGDClassifier supports partial_fit which allows incremental updates
without retraining from scratch. The gateway collects live request
features and every 1000 requests calls partial_fit to update the
model weights. This means:

- The model continuously adapts to the traffic patterns of this
  specific deployment environment
- New attack patterns that emerge after training are gradually learned
- The ensemble weight shifts toward SGDClassifier as it accumulates
  more live training data
- No expensive full retraining pipeline needed at runtime

IsolationForest does not support partial_fit which is why it is kept
as a static cold-start model while SGDClassifier handles adaptation.