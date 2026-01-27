import os
import numpy as np
import pandas as pd

from train.train_oraclead import train_oraclead
from infer.infer_scores import infer_scores
from metrics.evaluate import evaluate
from metrics.all_metrics import print_extended_metrics
from utils.postprocess import postprocess_predictions
from visualization.plot_heatmaps import plot_D_heatmap
from visualization.plot_timeseries import plot_timeseries_window



# =========================================================
# PATHS
# =========================================================
TRAIN_CSV = "data/ransyncoders/train.csv"
TEST_CSV = "data/ransyncoders/test.csv"
LABEL_CSV = "data/ransyncoders/test_label.csv"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE = 20


# =========================================================
# TRAIN
# =========================================================
print("\n===== TRAINING =====")

model, SLS, mean, std = train_oraclead(
    train_csv=TRAIN_CSV,
    window_size=WINDOW_SIZE,
    epochs=10
)


# =========================================================
# INFERENCE
# =========================================================
print("\n===== INFERENCE =====")

scores, labels = infer_scores(
    model=model,
    SLS=SLS,
    mean=mean,
    std=std,
    test_csv=TEST_CSV,
    label_csv=LABEL_CSV,
    window_size=WINDOW_SIZE,
    top_k=5,
    save_dir=RESULTS_DIR
)


# =========================================================
# PRIMARY EVALUATION (F1 + threshold)
# =========================================================
print("\n===== SEGMENT-LEVEL METRICS =====")

# This prints:
# Best Precision
# Best Recall
# Best F1-score
# Threshold
best_threshold = evaluate(scores, labels)


# =========================================================
# FREEZE DETECTOR (CRITICAL)
# =========================================================
print("\n===== FREEZING DETECTOR =====")
print(f"Using fixed threshold = {best_threshold:.4f}")

preds = postprocess_predictions(
    scores,
    best_threshold,
    min_len=5,
    gap=3,
    dilation=3
)


# =========================================================
# EXTENDED METRICS (CONSISTENT WITH F1)
# =========================================================
print_extended_metrics(
    preds=preds,
    scores=scores,
    labels=labels,
    window_size=WINDOW_SIZE,
    results_dir=RESULTS_DIR
)


# =========================================================
# VISUALIZATION — D-MATRIX HEATMAPS (FIGURE 3)
# =========================================================
print("\n===== PLOTTING D-MATRIX HEATMAPS =====")

D_files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.startswith("D_matrix_t")]
)

for f in D_files:
    t = f.split("t")[-1].split(".")[0]
    D = np.load(os.path.join(RESULTS_DIR, f))

    plot_D_heatmap(
        D,
        title=f"D-matrix at t={t}",
        save_path=os.path.join(RESULTS_DIR, f"heatmap_t{t}.png")
    )


# =========================================================
# VISUALIZATION — TIME-SERIES WINDOW (FIGURE 2)
# =========================================================
print("\n===== PLOTTING TIME-SERIES WINDOW =====")

df_test = pd.read_csv(TEST_CSV, engine="python")
X_raw = df_test.drop(columns=["timestamp_(min)"]).values

top_t = int(np.argmax(scores))
start = max(0, top_t - 100)
end = min(len(scores), top_t + 100)

variables = list(range(min(5, X_raw.shape[1])))

plot_timeseries_window(
    data=X_raw,
    scores=scores,
    labels=labels,
    variables=variables,
    start=start,
    end=end,
    save_path=os.path.join(RESULTS_DIR, "timeseries_top_anomaly.png")
)


# =========================================================
# DONE
# =========================================================
print("\n===== PIPELINE COMPLETE =====")
print(f"Results saved in: {RESULTS_DIR}/")

# from train.train_oraclead import train_oraclead
# from infer.infer_scores import infer_scores
# from metrics.evaluate import evaluate

# # -------------------------
# # PATHS
# # -------------------------
# TRAIN_CSV = "data/ransyncoders/train.csv"
# TEST_CSV = "data/ransyncoders/test.csv"
# LABEL_CSV = "data/ransyncoders/test_label.csv"

# # -------------------------
# # TRAIN
# # -------------------------
# model, SLS, mean, std = train_oraclead(
#     train_csv=TRAIN_CSV,
#     window_size=20,
#     epochs=10
# )

# # -------------------------
# # INFER
# # -------------------------
# scores, labels = infer_scores(
#     model=model,
#     SLS=SLS,
#     mean=mean,
#     std=std,
#     test_csv=TEST_CSV,
#     label_csv=LABEL_CSV,
#     window_size=20
# )

# # -------------------------
# # EVALUATE
# # -------------------------
# evaluate(scores, labels)
# import os
# import numpy as np
# import pandas as pd

# # -------------------------
# # IMPORTS
# # -------------------------
# from train.train_oraclead import train_oraclead
# from infer.infer_scores import infer_scores

# # Detection metrics (STRICT, main result)
# from metrics.evaluate import evaluate

# # Extended metrics (balanced, secondary)
# from metrics.all_metrics import print_all_metrics

# # Visualizations
# from visualization.plot_heatmaps import plot_D_heatmap
# from visualization.plot_timeseries import plot_timeseries_window


# # =========================================================
# # PATHS
# # =========================================================
# TRAIN_CSV = "data/ransyncoders/train.csv"
# TEST_CSV  = "data/ransyncoders/test.csv"
# LABEL_CSV = "data/ransyncoders/test_label.csv"

# RESULTS_DIR = "results"
# os.makedirs(RESULTS_DIR, exist_ok=True)


# # =========================================================
# # TRAIN
# # =========================================================
# print("\n===== TRAINING =====")

# model, SLS, mean, std = train_oraclead(
#     train_csv=TRAIN_CSV,
#     window_size=20,
#     epochs=10
# )


# # =========================================================
# # INFERENCE
# # =========================================================
# print("\n===== INFERENCE =====")

# scores, labels = infer_scores(
#     model=model,
#     SLS=SLS,
#     mean=mean,
#     std=std,
#     test_csv=TEST_CSV,
#     label_csv=LABEL_CSV,
#     window_size=20,
#     top_k=5,
#     save_dir=RESULTS_DIR
# )


# # =========================================================
# # PRIMARY EVALUATION (DO NOT TOUCH)
# # =========================================================
# print("\n===== EVALUATION (MAIN DETECTION METRICS) =====")
# evaluate(scores, labels)


# # =========================================================
# # SECONDARY EVALUATION (BALANCED, PAPER-STYLE)
# # =========================================================
# print("\n===== EVALUATION (EXTENDED METRICS) =====")
# print_all_metrics(scores, labels, results_dir=RESULTS_DIR)


# # =========================================================
# # VISUALIZATION — HEATMAPS (FIGURE 3)
# # =========================================================
# print("\n===== PLOTTING D-MATRIX HEATMAPS =====")

# D_files = sorted(
#     f for f in os.listdir(RESULTS_DIR) if f.startswith("D_matrix_t")
# )

# for f in D_files:
#     t = f.split("t")[-1].split(".")[0]
#     D = np.load(os.path.join(RESULTS_DIR, f))

#     plot_D_heatmap(
#         D,
#         title=f"D-matrix at t={t}",
#         save_path=os.path.join(RESULTS_DIR, f"heatmap_t{t}.png")
#     )


# # =========================================================
# # VISUALIZATION — TIME-SERIES (FIGURE 2)
# # =========================================================
# print("\n===== PLOTTING TIME-SERIES WINDOW =====")

# df_test = pd.read_csv(TEST_CSV, engine="python")
# X_raw = df_test.drop(columns=["timestamp_(min)"]).values

# # focus on strongest anomaly
# top_t = int(np.argmax(scores))
# start = max(0, top_t - 100)
# end   = min(len(scores), top_t + 100)

# variables = list(range(min(5, X_raw.shape[1])))

# plot_timeseries_window(
#     data=X_raw,
#     scores=scores,
#     labels=labels,
#     variables=variables,
#     start=start,
#     end=end,
#     save_path=os.path.join(RESULTS_DIR, "timeseries_top_anomaly.png")
# )


# # =========================================================
# # DONE
# # =========================================================
# print("\n===== PIPELINE COMPLETE =====")
# print(f"Results saved in: {RESULTS_DIR}/")

