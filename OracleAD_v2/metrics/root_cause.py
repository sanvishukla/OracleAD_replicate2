import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def variable_scores_from_D(D):
    """
    Compute variable-level deviation scores from D-matrix.

    Paper intuition:
    Variables with large row/column sums contribute more.
    """
    # sum over rows (or columns, same since symmetric)
    return D.sum(axis=1)


def compute_variable_metrics(D_matrices, gt_variables, top_k=None):
    """
    Compute V-ROC and V-PR.

    Parameters
    ----------
    D_matrices : list or np.ndarray
        Shape (T, V, V)
    gt_variables : list of lists
        gt_variables[t] = list of ground-truth anomalous variables at time t
    top_k : int or None
        If set, only evaluate top-k anomaly timesteps

    Returns
    -------
    vroc, vpr
    """

    all_scores = []
    all_labels = []

    for t, D in enumerate(D_matrices):
        scores = variable_scores_from_D(D)

        labels = np.zeros_like(scores)
        labels[gt_variables[t]] = 1

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # handle edge cases
    if len(np.unique(all_labels)) < 2:
        return np.nan, np.nan

    vroc = roc_auc_score(all_labels, all_scores)
    vpr = average_precision_score(all_labels, all_scores)

    return vroc, vpr
