import numpy as np

def moving_average(scores, window=200):
    """
    Smooth anomaly scores using moving average.

    Parameters
    ----------
    scores : np.ndarray
        Shape (T,)
    window : int

    Returns
    -------
    smoothed_scores : np.ndarray
    """
    if window <= 1:
        return scores

    kernel = np.ones(window) / window
    return np.convolve(scores, kernel, mode="same")


def normalize(scores):
    """
    Min-max normalize scores to [0, 1].
    """
    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)
