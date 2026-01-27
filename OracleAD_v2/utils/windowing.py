import numpy as np

def make_windows(data, labels=None, window_size=10):
    """
    Create sliding windows from multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, D)
    labels : np.ndarray or None
        Shape (T,)
    window_size : int

    Returns
    -------
    X : np.ndarray
        Shape (N, window_size, D)
    y : np.ndarray or None
        Labels aligned with last timestep of each window
    """

    X = []
    y = []

    T = data.shape[0]

    for i in range(T - window_size + 1):
        X.append(data[i : i + window_size])

        if labels is not None:
            y.append(labels[i + window_size - 1])

    X = np.asarray(X)

    if labels is not None:
        y = np.asarray(y)
        return X, y

    return X
