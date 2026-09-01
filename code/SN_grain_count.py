# This script evaluates the normalized grain-count evolution predicted by the trained models.
# It generates per-sequence and global plots, CSV files, and evaluation metrics.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import shutil
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import gc
import csv

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl

# mpl.rcParams.update({
#     "figure.dpi": 600,
#     "savefig.dpi": 600,
#     "font.size": 18,
#     "axes.titlesize": 15,
#     "axes.labelsize": 17,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "legend.fontsize": 14,
#     "figure.autolayout": True,
#     "lines.linewidth": 3.5,
# })

mpl.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 18,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.autolayout": True,
    "lines.linewidth": 2.5,
})

# ==========================
# CONFIGURATION
# ==========================
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

base_output_dir = "/media/admin-eyounes/T7/1-Article/github_code/SN_output/1heure"
new_sequences_root = "/media/admin-eyounes/T7/1-Article/github_code/test_sequences"

# base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/SN_output/1heure"
# new_sequences_root = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/test_sequences"

datasize = 30
window_size = 5
output_size = 55

AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types = ["rnn", "lstm", "tcn", "transformer"]

USE_MINUTES = False
DT_MIN = 1.0

out_root = os.path.join(base_output_dir, "SN_grain_count_over_time_new_sequences")

colors = {
    "gt": "#444444",
    "rnn": "green",
    "lstm": "red",
    "tcn": "orange",
    "transformer": "blue",
}

labels = {
    "gt": "Ground truth normalized",
    "rnn": "Model RNN",
    "lstm": "Model LSTM",
    "tcn": "Model TCN",
    "transformer": "Model TRANSFORMER",
}

# ==========================
# UTILITY FUNCTIONS
# ==========================
def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def numerical_sort(value):
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p for p in parts]

# ==========================
# DATA LOADING
# ==========================
def load_one_sequence_in_memory(sequence_dir: str):
    """
    Returns:
      seq_norm: (T, datasize) normalized sequence
      sums:     (T,) actual sums, retained but not used here
      files:    list of files
    """
    seq = []
    sums = []
    files = []

    for file_name in sorted(os.listdir(sequence_dir), key=numerical_sort):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(sequence_dir, file_name)

        with open(file_path, "r") as f:
            next(f)
            vals = f.read().strip().split()
            vals = [float(v) for v in vals]

        if len(vals) != datasize:
            raise ValueError(
                f"Fichier {file_path}: attendu {datasize} valeurs, reçu {len(vals)}"
            )

        s = float(sum(vals))
        sums.append(s)

        if s != 0:
            norm = [round(v / s, 16) for v in vals]
        else:
            norm = [0.0 for _ in vals]

        seq.append(norm)
        files.append(file_name)

    if len(seq) == 0:
        return None, None, None

    return np.array(seq, dtype=float), np.array(sums, dtype=float), files

# ==========================
# NORMALIZED PREDICTION
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    The model receives normalized data and predicts normalized data.
    No denormalization is performed here.
    """
    current_window = seq_norm[:window_size].copy()
    preds = []

    for _ in range(output_size):
        inp = current_window[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0]

        pred = np.maximum(pred, 0)

        preds.append(pred)

        pred_sum = np.sum(pred) + 1e-8
        pred_norm_for_next = pred / pred_sum

        current_window = np.vstack((current_window[1:], pred_norm_for_next))

    return np.array(preds, dtype=float)

def predict_normalized_one_sequence(model, seq_norm, window_size, output_size):
    """
    Returns:
      preds_norm: normalized model predictions
      y_norm:      normalized ground-truth values
    """
    preds_norm = autoregressive_predict_norm(
        model,
        seq_norm,
        window_size,
        output_size
    )

    y_norm = seq_norm[window_size:window_size + output_size]

    return preds_norm, y_norm

# ==========================
# NORMALIZED GRAIN COUNT AND METRICS
# ==========================
def grain_count_from_distribution(dist_2d):
    """
    dist_2d: (T, datasize)
    Returns the sum of the bins at each time step.
    For normalized data, the ground-truth value is normally close to 1.
    """
    return np.sum(dist_2d, axis=1)

def compute_metrics_1d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    diff = y_pred - y_true

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    mask = y_true != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mre = 0.0

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MRE%": mre,
        "R2": r2,
    }

def print_metrics(title, metrics):
    print(
        f"{title}: "
        f"MAE={metrics['MAE']:.6f} | "
        f"MSE={metrics['MSE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"MRE={metrics['MRE%']:.4f}% | "
        f"R²={metrics['R2']:.6f}"
    )

# ==========================
# PLOTS AND CSV FILES
# ==========================
def get_x_axis(window_size, output_size, use_minutes=False, dt_min=1.0):
    steps = np.arange(window_size + 1, window_size + 1 + output_size)

    if use_minutes:
        return (steps - (window_size + 1)) * dt_min

    return steps

def save_sequence_grain_count_plot_and_csv(seq_name, x, gt_curve, pred_curves_by_model, out_dir):
    safe_makedirs(out_dir)

    csv_path = os.path.join(out_dir, "normalized_grain_count_over_time_ALL_MODELS.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["x", "GT_NORMALIZED"] + [
            mt.upper() for mt in model_types if mt in pred_curves_by_model
        ]
        w.writerow(header)

        for i in range(len(x)):
            row = [float(x[i]), float(gt_curve[i])]

            for mt in model_types:
                if mt in pred_curves_by_model:
                    row.append(float(pred_curves_by_model[mt][i]))

            w.writerow(row)
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8,6))
    plt.plot(
        x,
        gt_curve,
        marker="o",
        label=labels["gt"],
        color=colors["gt"]
    )

    for mt in model_types:
        if mt in pred_curves_by_model:
            plt.plot(
                x,
                pred_curves_by_model[mt],
                marker="o",
                label=labels.get(mt, mt.upper()),
                color=colors.get(mt, None)
            )

    plt.xlabel("Time (min)" if USE_MINUTES else "Time step")
    plt.ylabel("Sum of grain-size frequencies")
    plt.title("")
    # plt.title("Evolution of the Sum of Grain-Size Frequencies")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "normalized_grain_count_over_time_ALL_MODELS.png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[SEQ {seq_name}] saved: {csv_path} + {fig_path}")

def save_global_mean_plot_and_csv(x, gt_mean, mean_by_model, out_dir):
    safe_makedirs(out_dir)

    csv_path = os.path.join(out_dir, "normalized_grain_count_over_time_MEAN_ALL_MODELS.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["x", "GT_NORMALIZED_MEAN"] + [
            f"{mt.upper()}_MEAN" for mt in model_types if mt in mean_by_model
        ]
        w.writerow(header)

        for i in range(len(x)):
            row = [float(x[i]), float(gt_mean[i])]

            for mt in model_types:
                if mt in mean_by_model:
                    row.append(float(mean_by_model[mt][i]))

            w.writerow(row)
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 6))  
    plt.plot(
        x,
        gt_mean,
        marker="o",
        label="GT NORMALIZED MEAN",
        color=colors["gt"]
    )

    for mt in model_types:
        if mt in mean_by_model:
            plt.plot(
                x,
                mean_by_model[mt],
                marker="o",
                label=f"{mt.upper()} MEAN",
                color=colors.get(mt, None)
            )

    plt.xlabel("Time (min)" if USE_MINUTES else "Time step")
    plt.ylabel("Sum of grain-size frequencies")
    plt.title("")
    # plt.title("Evolution of the Sum of Grain-Size Frequencies")
    # plt.ylabel("Mean Normalized Grain Count")
    # plt.title("Mean Normalized Grain Count Evolution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "normalized_grain_count_over_time_MEAN_ALL_MODELS.png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[GLOBAL MEAN] saved: {csv_path} + {fig_path}")

# ==========================
# MAIN
# ==========================
def main():
    print("=" * 80)
    print("NORMALIZED GRAIN COUNT OVER TIME — NEW SEQUENCES ONLY")
    print("=" * 80)

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}")

    if not os.path.exists(new_sequences_root):
        raise FileNotFoundError(
            f"Dossier new_sequences_root introuvable: {new_sequences_root}"
        )

    reset_directory(out_root)

    seq_names = []
    seq_paths = []

    for seq_name in sorted(os.listdir(new_sequences_root), key=numerical_sort):
        sp = os.path.join(new_sequences_root, seq_name)

        if os.path.isdir(sp):
            seq_names.append(seq_name)
            seq_paths.append(sp)

    if len(seq_names) == 0:
        raise RuntimeError("Aucune séquence trouvée dans new_sequences_root.")

    models = {}

    for mt in model_types:
        model_path = os.path.join(base_output_dir, f"model_{mt}.keras")

        if not os.path.exists(model_path):
            print(f"[WARN] modèle manquant, ignoré: {model_path}")
            continue

        models[mt] = tf.keras.models.load_model(model_path)
        print(f"✅ Modèle chargé: {model_path}")

    if len(models) == 0:
        raise RuntimeError(
            "Aucun modèle chargé. Vérifie model_<type>.keras dans base_output_dir."
        )

    gt_curves_all = []
    pred_curves_all_by_model = {mt: [] for mt in models.keys()}
    global_metrics_by_model = {mt: [] for mt in models.keys()}

    x = get_x_axis(
        window_size,
        output_size,
        use_minutes=USE_MINUTES,
        dt_min=DT_MIN
    )

    print("\n" + "=" * 80)
    print("[PER SEQUENCE] Normalized Grain Count plots + metrics")
    print("=" * 80)

    for seq_name, seq_path in zip(seq_names, seq_paths):
        seq_norm, sums, files = load_one_sequence_in_memory(seq_path)

        if seq_norm is None:
            print(f"[SKIP] {seq_name}: vide.")
            continue

        if len(seq_norm) < window_size + output_size:
            print(
                f"[SKIP] {seq_name}: trop courte ({len(seq_norm)}), "
                f"min={window_size + output_size}"
            )
            continue

        # Normalized ground truth
        gt_dist = seq_norm[window_size:window_size + output_size]
        gt_curve = grain_count_from_distribution(gt_dist)

        pred_curves_by_model = {}

        for mt, model in models.items():
            preds_norm, y_norm = predict_normalized_one_sequence(
                model,
                seq_norm,
                window_size,
                output_size
            )

            pred_curve = grain_count_from_distribution(preds_norm)
            pred_curves_by_model[mt] = pred_curve

            met = compute_metrics_1d(gt_curve, pred_curve)
            global_metrics_by_model[mt].append(met)

            print_metrics(f"[{seq_name}][{mt.upper()}]", met)

        seq_out = os.path.join(out_root, "per_sequence", seq_name)

        save_sequence_grain_count_plot_and_csv(
            seq_name,
            x,
            gt_curve,
            pred_curves_by_model,
            seq_out
        )

        gt_curves_all.append(gt_curve)

        for mt in models.keys():
            if mt in pred_curves_by_model:
                pred_curves_all_by_model[mt].append(pred_curves_by_model[mt])

        gc.collect()

    if len(gt_curves_all) == 0:
        raise RuntimeError(
            "Aucune séquence valide pour calculer les courbes normalisées."
        )

    gt_curves_all = np.stack(gt_curves_all, axis=0)
    gt_mean = np.mean(gt_curves_all, axis=0)

    mean_by_model = {}

    for mt, curves_list in pred_curves_all_by_model.items():
        if len(curves_list) == 0:
            continue

        arr = np.stack(curves_list, axis=0)
        mean_by_model[mt] = np.mean(arr, axis=0)

    global_out = os.path.join(out_root, "global")

    save_global_mean_plot_and_csv(
        x,
        gt_mean,
        mean_by_model,
        global_out
    )

    summary_csv = os.path.join(
        global_out,
        "GLOBAL_METRICS_NORMALIZED_GRAIN_COUNT_BY_MODEL.csv"
    )

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "model_type",
            "mean_MAE",
            "mean_MSE",
            "mean_RMSE",
            "mean_MRE_percent",
            "mean_R2"
        ])

        for mt in model_types:
            if mt not in global_metrics_by_model or len(global_metrics_by_model[mt]) == 0:
                continue

            mets = global_metrics_by_model[mt]

            w.writerow([
                mt,
                float(np.mean([m["MAE"] for m in mets])),
                float(np.mean([m["MSE"] for m in mets])),
                float(np.mean([m["RMSE"] for m in mets])),
                float(np.mean([m["MRE%"] for m in mets])),
                float(np.mean([m["R2"] for m in mets])),
            ])

    print("\n" + "=" * 80)
    print("[GLOBAL] Summary metrics saved:")
    print(f"  - {summary_csv}")
    print("=" * 80)

    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 80)
    print("FIN. Sorties générées:")
    print(f"  - {out_root}/per_sequence/<seq>/normalized_grain_count_over_time_ALL_MODELS.png + .csv")
    print(f"  - {out_root}/global/normalized_grain_count_over_time_MEAN_ALL_MODELS.png + .csv")
    print(f"  - {out_root}/global/GLOBAL_METRICS_NORMALIZED_GRAIN_COUNT_BY_MODEL.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()