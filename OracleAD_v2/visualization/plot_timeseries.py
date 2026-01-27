import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries_window(
    data,
    scores,
    labels,
    variables,
    start,
    end,
    save_path=None
):
    """
    Plot raw variables + anomaly score for a time window.
    """

    t = np.arange(start, end)

    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars + 1, 1,
        figsize=(10, 2 * (n_vars + 1)),
        sharex=True
    )

    for i, var_idx in enumerate(variables):
        axes[i].plot(t, data[start:end, var_idx])
        axes[i].set_ylabel(f"Var {var_idx}")

        # shade anomaly region
        for j in range(start, end):
            if labels[j] == 1:
                axes[i].axvspan(j, j + 1, color="red", alpha=0.2)

    axes[-1].plot(t, scores[start:end], color="red")
    axes[-1].set_ylabel("Anomaly Score")
    axes[-1].set_xlabel("Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/ransyncoders/test.csv")
    labels = np.load("results/labels.npy")
    scores = np.load("results/anomaly_scores.npy")

    X = df.drop(columns=["timestamp_(min)"]).values

    plot_timeseries_window(
        data=X,
        scores=scores,
        labels=labels,
        variables=[8, 12, 13],
        start=14900,
        end=15000,
        save_path="results/timeseries_example.png"
    )
