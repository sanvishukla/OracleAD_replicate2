import numpy as np
import matplotlib.pyplot as plt
import os


def plot_D_heatmap(D, title, save_path=None):
    """
    Plot a deviation matrix heatmap (Figure-3 style).
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="inferno", aspect="auto")
    plt.colorbar(label="Deviation Intensity")
    plt.xlabel("Variable Index")
    plt.ylabel("Variable Index")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    results_dir = "results"

    files = sorted([f for f in os.listdir(results_dir) if f.startswith("D_matrix")])

    for f in files:
        t = f.split("t")[-1].split(".")[0]
        D = np.load(os.path.join(results_dir, f))

        plot_D_heatmap(
            D,
            title=f"D-matrix at t={t}",
            save_path=os.path.join(results_dir, f"heatmap_t{t}.png")
        )
