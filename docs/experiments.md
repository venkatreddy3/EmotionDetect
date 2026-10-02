# Experiment Tracking and Model Evaluation

## Benchmark & Target Evaluation Metrics

To ensure academic rigor and transparent evaluation without fabricating unverified numbers, our evaluation protocol uses the following primary benchmarks:

1. **Overall Classification Accuracy** (Top-1 Accuracy on balanced and full test sets).
2. **Per-Class Precision, Recall, and Macro-Averaged F1-Score** (Crucial due to class imbalances in FER datasets).
3. **Confusion Matrix Analysis** (Tracking common misclassifications, e.g., Fear vs Surprise, Neutral vs Sad).
4. **Inference Latency (ms)**:
   - CPU benchmark (x86_64 and ARM).
   - End-to-end pipeline latency (Face Detection + Preprocessing + Forward Pass).
5. **Model Size and Parameter Count**: Targeting under 15MB for fast edge and cloud deployment.

---

## Model Evolution Matrix

| Model Identifier | Architecture Description | Parameters | Test Accuracy | Macro F1 | Status |
|---|---|---|---|---|---|
| `exp-001-baseline` | 4-Block Conv2D + MaxPool + Dropout | TBD | TBD | TBD | Planned |
| `exp-002-minixception` | Residual Depthwise Separable Conv | TBD | TBD | TBD | Planned |
| `exp-003-augmented` | ResNet-style + CLAHE Normalization | TBD | TBD | TBD | Planned |

*(Evaluation metrics will be computed systematically using `training/evaluate.py` and populated automatically into `artifacts/reports/` upon experimental execution.)*
