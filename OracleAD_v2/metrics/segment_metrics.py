import numpy as np


def extract_segments(labels):
    """
    Extract contiguous anomaly segments from binary labels.

    Returns list of (start, end) indices.
    """
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_segment:
            in_segment = True
            start = i
        elif val == 0 and in_segment:
            segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(labels) - 1))

    return segments


def segment_f1(scores, labels, threshold):
    """
    Segment-level F1 score.

    A predicted segment is correct if it overlaps
    with any true anomaly segment.
    """

    pred_labels = (scores >= threshold).astype(int)

    true_segments = extract_segments(labels)
    pred_segments = extract_segments(pred_labels)

    if len(pred_segments) == 0:
        return 0.0, 0.0, 0.0

    matched_true = set()
    tp = 0

    for ps in pred_segments:
        for i, ts in enumerate(true_segments):
            # check overlap
            if ps[1] >= ts[0] and ps[0] <= ts[1]:
                tp += 1
                matched_true.add(i)
                break

    fp = len(pred_segments) - tp
    fn = len(true_segments) - len(matched_true)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1
