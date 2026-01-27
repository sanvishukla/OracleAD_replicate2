# import numpy as np
# import pandas as pd
# import torch

# from utils.windowing import make_windows
# from utils.smoothing import moving_average, normalize
# from train.train_oraclead import OracleAD


# def infer_scores(
#     model,
#     SLS,
#     mean,
#     std,
#     test_csv,
#     label_csv,
#     window_size=20,
#     device="cpu"
# ):
#     """
#     Runs inference and returns:
#     - smoothed anomaly scores
#     - aligned ground-truth labels
#     """

#     # -------------------------
#     # Load test data
#     # -------------------------
#     df_test = pd.read_csv(test_csv)
#     df_label = pd.read_csv(label_csv)

#     X_raw = df_test.drop(columns=["timestamp_(min)"]).values
#     labels = df_label["label"].values

#     # normalize using TRAIN stats
#     X_raw = (X_raw - mean) / std
#     X_raw = np.nan_to_num(X_raw)

#     # -------------------------
#     # Windowing
#     # -------------------------
#     X, y = make_windows(
#         X_raw,
#         labels=labels,
#         window_size=window_size
#     )

#     X = torch.tensor(X, dtype=torch.float32).to(device)

#     model.eval()

#     pred_errors = []
#     latent_states = []

#     # -------------------------
#     # Inference loop
#     # -------------------------
#     with torch.no_grad():
#         for i in range(len(X)):
#             x = X[i].unsqueeze(0)

#             target = x[:, -1, :]
#             y_hat, h = model(x)

#             # prediction error (L1 is better for anomalies)
#             err = torch.mean(torch.abs(y_hat - target)).item()
#             pred_errors.append(err)

#             latent_states.append(h.squeeze(0).cpu().numpy())

#     pred_errors = np.array(pred_errors)
#     latent_states = np.array(latent_states)

#     # -------------------------
#     # Structural deviation score
#     # -------------------------
#     deviations = []

#     for h in latent_states:
#         diff = h - SLS.mean(axis=1)
#         dev = np.linalg.norm(diff)
#         deviations.append(dev)

#     deviations = np.array(deviations)

#     # -------------------------
#     # NORMALIZE + SMOOTH (CRITICAL)
#     # -------------------------
#     P = normalize(pred_errors)
#     D = normalize(deviations)

#     # smooth prediction score (paper-style)
#     P_smooth = moving_average(P, window=200)

#     # final anomaly score (weighted, stable)
#     A = 0.7 * P_smooth + 0.3 * D

#     A = normalize(A)

#     return A, y
import numpy as np
import pandas as pd
import torch
import os

from utils.windowing import make_windows
from utils.smoothing import moving_average, normalize
from train.train_oraclead import OracleAD


def infer_scores(
    model,
    SLS,
    mean,
    std,
    test_csv,
    label_csv,
    window_size=20,
    device="cpu",
    top_k=5,
    save_dir="results"
):
    """
    Runs inference and returns:
    - anomaly scores
    - labels
    - saves D-matrices for top-K anomaly timestamps
    """

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Load test data (ROBUST CSV LOADING)
    # --------------------------------------------------
    print(f"Loading test CSV from: {os.path.abspath(test_csv)}")
    print(f"Test CSV exists: {os.path.exists(test_csv)}")

    df_test = pd.read_csv(
        test_csv,
        engine="python"   # <-- IMPORTANT FIX for macOS timeouts
    )

    print(f"Loading label CSV from: {os.path.abspath(label_csv)}")
    print(f"Label CSV exists: {os.path.exists(label_csv)}")

    df_label = pd.read_csv(
        label_csv,
        engine="python"
    )

    X_raw = df_test.drop(columns=["timestamp_(min)"]).values
    labels = df_label["label"].values

    # --------------------------------------------------
    # Normalize using TRAIN statistics
    # --------------------------------------------------
    X_raw = (X_raw - mean) / std
    X_raw = np.nan_to_num(X_raw)

    # --------------------------------------------------
    # Windowing
    # --------------------------------------------------
    X, y = make_windows(
        X_raw,
        labels=labels,
        window_size=window_size
    )

    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()

    pred_errors = []
    latent_states = []

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    with torch.no_grad():
        for i in range(len(X)):
            x = X[i].unsqueeze(0)

            target = x[:, -1, :]
            y_hat, h = model(x)

            # L1 prediction error
            err = torch.mean(torch.abs(y_hat - target)).item()
            pred_errors.append(err)

            latent_states.append(h.squeeze(0).cpu().numpy())

    pred_errors = np.array(pred_errors)
    latent_states = np.array(latent_states)

    # --------------------------------------------------
    # Structural deviation (D-matrix + scalar deviation)
    # --------------------------------------------------
    mu = latent_states.mean(axis=0)

    deviations = []
    D_matrices = []

    for h in latent_states:
        diff = h - mu

        # OracleAD deviation matrix
        D_t = np.outer(diff, diff)
        D_matrices.append(D_t)

        deviations.append(np.linalg.norm(diff))

    deviations = np.array(deviations)
    D_matrices = np.array(D_matrices)

    # --------------------------------------------------
    # Normalize + smooth (CRITICAL for F1)
    # --------------------------------------------------
    P = normalize(pred_errors)
    D = normalize(deviations)

    P_smooth = moving_average(P, window=200)

    # Final anomaly score
    A = 0.7 * P_smooth + 0.3 * D
    A = normalize(A)

    # --------------------------------------------------
    # Save TOP-K D-matrices
    # --------------------------------------------------
    top_indices = np.argsort(A)[-top_k:][::-1]

    print("\nTop-K anomaly timestamps:")
    for idx in top_indices:
        print(f"  t={idx}, score={A[idx]:.4f}")

        np.save(
            os.path.join(save_dir, f"D_matrix_t{idx}.npy"),
            D_matrices[idx]
        )

    # Save scores & labels
    np.save(os.path.join(save_dir, "anomaly_scores.npy"), A)
    np.save(os.path.join(save_dir, "labels.npy"), y)

    return A, y
