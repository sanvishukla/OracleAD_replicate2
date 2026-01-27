import numpy as np

from metrics.segment_metrics import segment_f1
from utils.smoothing import normalize


def evaluate(scores, labels):
    """
    Segment-level evaluation using percentile thresholds.
    """

    scores = normalize(scores)

    percentiles = np.linspace(90, 99.9, 50)

    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_t = 0.0

    for p in percentiles:
        t = np.percentile(scores, p)
        precision, recall, f1 = segment_f1(scores, labels, t)

        if f1 > best_f1:
            best_f1 = f1
            best_p = precision
            best_r = recall
            best_t = t

    print("\n=== SEGMENT-LEVEL METRICS ===")
    print(f"Best Precision : {best_p:.4f}")
    print(f"Best Recall    : {best_r:.4f}")
    print(f"Best F1-score  : {best_f1:.4f}")
    print(f"Threshold      : {best_t:.4f}")

    return best_f1
