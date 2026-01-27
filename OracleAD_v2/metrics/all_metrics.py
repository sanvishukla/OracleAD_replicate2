import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_segments(binary):
    segments = []
    in_seg = False
    start = 0

    for i, v in enumerate(binary):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False

    if in_seg:
        segments.append((start, len(binary) - 1))

    return segments


def align_labels_to_windows(labels, window_size):
    aligned = []
    for i in range(len(labels) - window_size + 1):
        aligned.append(int(labels[i:i + window_size].max() > 0))
    return np.array(aligned)


# ---------------------------------------------------------
# R-F1 (range based)
# ---------------------------------------------------------
def range_f1(preds, labels):
    gt_segs = extract_segments(labels)
    pr_segs = extract_segments(preds)

    if len(gt_segs) == 0 or len(pr_segs) == 0:
        return 0.0

    overlaps = []

    for gs in gt_segs:
        g_len = gs[1] - gs[0] + 1
        best = 0.0
        for ps in pr_segs:
            ov = max(0, min(gs[1], ps[1]) - max(gs[0], ps[0]) + 1)
            best = max(best, ov / g_len)
        overlaps.append(best)

    recall = np.mean(overlaps)
    precision = recall
    return 2 * precision * recall / (precision + recall + 1e-8)


# ---------------------------------------------------------
# A-ROC / A-PR (event level)
# ---------------------------------------------------------
def anomaly_level_metrics(scores, labels):
    gt_segs = extract_segments(labels)

    seg_scores = []
    seg_labels = []

    for s, e in gt_segs:
        seg_scores.append(scores[s:e + 1].max())
        seg_labels.append(1)

    normal_scores = scores[labels == 0]
    seg_scores.extend(normal_scores.tolist())
    seg_labels.extend([0] * len(normal_scores))

    return (
        roc_auc_score(seg_labels, seg_scores),
        average_precision_score(seg_labels, seg_scores),
    )


# ---------------------------------------------------------
# V-ROC / V-PR (root-cause)
# ---------------------------------------------------------
def variable_level_metrics(results_dir):
    files = [f for f in os.listdir(results_dir) if f.startswith("D_matrix_t")]

    all_scores = []
    all_labels = []

    for f in files:
        D = np.load(os.path.join(results_dir, f))
        scores = D.sum(axis=1)

        k = max(1, int(0.1 * len(scores)))
        labels = np.zeros_like(scores)
        labels[np.argsort(scores)[-k:]] = 1

        all_scores.extend(scores)
        all_labels.extend(labels)

    return (
        roc_auc_score(all_labels, all_scores),
        average_precision_score(all_labels, all_scores),
    )


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
def print_extended_metrics(
    preds,
    scores,
    labels,
    window_size=20,
    results_dir="results"
):
    labels = align_labels_to_windows(labels, window_size)
    preds = preds[:len(labels)]
    scores = scores[:len(labels)]

    rf1 = range_f1(preds, labels)
    aroc, apr = anomaly_level_metrics(scores, labels)
    vroc, vpr = variable_level_metrics(results_dir)

    print("\n===== EVALUATION (EXTENDED METRICS) =====")
    print(f"R-F1    : {rf1:.4f}")
    print("Aff-F1  : N/A (not implemented)")
    print(f"A-ROC   : {aroc:.4f}")
    print(f"A-PR    : {apr:.4f}")
    print(f"V-ROC   : {vroc:.4f}")
    print(f"V-PR    : {vpr:.4f}")
