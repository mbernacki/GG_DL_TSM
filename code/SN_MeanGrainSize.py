# This script computes the mean grain size over time from normalized distributions.
# It displays the ground truth from t=0 and model predictions from t=window_size.

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
import matplotlib as mpl
from sklearn.metrics import r2_score


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

# Bin centers
# If the actual physical bin centers are available, replace this line.
# BIN_CENTERS = np.arange(1, datasize + 1, dtype=float)


# Actual bins distributed between 0 and 0.17 mm
bins = np.linspace(0, 0.17, datasize + 1)
BIN_CENTERS = 0.5 * (bins[:-1] + bins[1:])


# Output directory
out_root = os.path.join(
    base_output_dir,
    "SN_mean_grain_size_over_time_NORMALIZED_GT_FROM_ZERO"
)

colors = {
    "gt": "#444444",
    "rnn": "green",
    "lstm": "red",
    "tcn": "orange",
    "transformer": "blue",
}

labels = {
    "gt": "Ground truth",
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
      seq_norm: normalized distribution with shape (T, datasize)
      sums: actual sums with shape (T,), retained but not used
      files: list of files
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
# AUTOREGRESSIVE PREDICTION
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    Performs autoregressive prediction on normalized distributions.

    The model receives the first window_size time steps
    and then predicts output_size future time steps.

    Returns:
      preds_norm: array with shape (output_size, datasize)
    """
    current_window = seq_norm[:window_size].copy()
    preds = []

    for _ in range(output_size):
        inp = current_window[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0]

        # Prevent negative values
        pred = np.maximum(pred, 0)

        # Renormalize the prediction
        pred_sum = np.sum(pred) + 1e-8
        pred_norm = pred / pred_sum

        preds.append(pred_norm)

        current_window = np.vstack((current_window[1:], pred_norm))

    return np.array(preds, dtype=float)


# ==========================
# MEAN GRAIN SIZE AND METRICS
# ==========================
def mean_grain_size_from_distribution(dist_2d, bin_centers):
    """
    Computes:
      Mean Grain Size = sum(distribution * bin_center) / sum(distribution)

    This function works with normalized and non-normalized distributions.
    """
    dist_2d = np.asarray(dist_2d, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)

    denom = np.sum(dist_2d, axis=1)
    numer = dist_2d @ bin_centers

    mean_size = np.full_like(denom, fill_value=np.nan, dtype=float)

    mask = denom != 0
    mean_size[mask] = numer[mask] / denom[mask]

    return mean_size


def compute_metrics_1d(y_true, y_pred):
    """
    y_true and y_pred have shape (T,).

    Returns:
      MAE, MSE, RMSE, MRE%, and R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(valid):
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MRE%": np.nan,
            "R2": np.nan,
        }

    yt = y_true[valid]
    yp = y_pred[valid]
    diff = yp - yt

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    mask = yt != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)
    else:
        mre = 0.0

    try:
        r2 = float(r2_score(yt, yp))
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
        f"MAE={metrics['MAE']:.4f} | "
        f"MSE={metrics['MSE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"MRE={metrics['MRE%']:.2f}% | "
        f"R²={metrics['R2']:.4f}"
    )


# ==========================
# X-AXIS
# ==========================
def get_x_axis_full(length, use_minutes=False, dt_min=1.0):
    steps = np.arange(length)

    if use_minutes:
        return steps * dt_min

    return steps


# ==========================
# PLOTS AND CSV FILES
# ==========================
def save_sequence_mean_size_plot_and_csv(
    seq_name,
    x_full,
    gt_curve_full,
    pred_curves_full_by_model,
    out_dir
):
    safe_makedirs(out_dir)

    csv_path = os.path.join(
        out_dir,
        "mean_grain_size_over_time_ALL_MODELS.csv"
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        header = ["x", "GT"] + [
            mt.upper()
            for mt in model_types
            if mt in pred_curves_full_by_model
        ]

        w.writerow(header)

        for i in range(len(x_full)):
            row = [
                float(x_full[i]),
                "" if not np.isfinite(gt_curve_full[i]) else float(gt_curve_full[i])
            ]

            for mt in model_types:
                if mt in pred_curves_full_by_model:
                    v = pred_curves_full_by_model[mt][i]
                    row.append("" if not np.isfinite(v) else float(v))

            w.writerow(row)

    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_full,
        gt_curve_full,
        marker="o",
        label=labels["gt"],
        color=colors["gt"]
    )

    for mt in model_types:
        if mt in pred_curves_full_by_model:
            plt.plot(
                x_full,
                pred_curves_full_by_model[mt],
                marker="o",
                label=labels.get(mt, mt.upper()),
                color=colors.get(mt, None)
            )

    plt.axvline(
        x=x_full[window_size],
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        color="black"
    )

    plt.xlabel("Time (min)" if USE_MINUTES else "Time step")
    # plt.ylabel("Mean Grain Size")
    plt.ylabel("Mean Grain Size (mm)")
    # plt.title("Mean Grain Size Evolution")
    plt.title("")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(
        out_dir,
        "mean_grain_size_over_time_ALL_MODELS.png"
    )

    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[SEQ {seq_name}] saved: {csv_path} + {fig_path}")


def save_global_mean_plot_and_csv(
    x_full,
    gt_mean_full,
    mean_full_by_model,
    out_dir
):
    safe_makedirs(out_dir)

    csv_path = os.path.join(
        out_dir,
        "mean_grain_size_over_time_MEAN_ALL_MODELS.csv"
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        header = ["x", "GT_MEAN"] + [
            f"{mt.upper()}_MEAN"
            for mt in model_types
            if mt in mean_full_by_model
        ]

        w.writerow(header)

        for i in range(len(x_full)):
            row = [
                float(x_full[i]),
                "" if not np.isfinite(gt_mean_full[i]) else float(gt_mean_full[i])
            ]

            for mt in model_types:
                if mt in mean_full_by_model:
                    v = mean_full_by_model[mt][i]
                    row.append("" if not np.isfinite(v) else float(v))

            w.writerow(row)

    plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(12, 8))
    plt.plot(
        x_full,
        gt_mean_full,
        marker="o",
        label="GT MEAN",
        color=colors["gt"]
    )

    for mt in model_types:
        if mt in mean_full_by_model:
            plt.plot(
                x_full,
                mean_full_by_model[mt],
                marker="o",
                label=f"{mt.upper()} MEAN",
                color=colors.get(mt, None)
            )

    plt.axvline(
        x=x_full[window_size],
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        color="black"
    )

    plt.xlabel("Time (min)" if USE_MINUTES else "Time step")
    plt.ylabel("Mean Grain Size (mm)")
    plt.title("")
    # plt.title("Mean Grain Size Evolution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(
        out_dir,
        "mean_grain_size_over_time_MEAN_ALL_MODELS.png"
    )

    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[GLOBAL MEAN] saved: {csv_path} + {fig_path}")


# ==========================
# MAIN
# ==========================
def main():
    print("=" * 80)
    print("SCRIPT — MEAN GRAIN SIZE NORMALIZED — GT FROM ZERO")
    print("=" * 80)

    if len(BIN_CENTERS) != datasize:
        raise ValueError(
            f"BIN_CENTERS doit avoir {datasize} valeurs, reçu {len(BIN_CENTERS)}"
        )

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}"
            )

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
        raise RuntimeError(
            "Aucune séquence trouvée dans new_sequences_root."
        )

    # Load the models
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

    gt_curves_full_all = []
    pred_curves_full_all_by_model = {mt: [] for mt in models.keys()}
    global_metrics_by_model = {mt: [] for mt in models.keys()}

    print("\n" + "=" * 80)
    print("[PER SEQUENCE] Mean Grain Size plots + metrics")
    print("=" * 80)

    for seq_name, seq_path in zip(seq_names, seq_paths):
        seq_norm, sums, files = load_one_sequence_in_memory(seq_path)

        if seq_norm is None:
            print(f"[SKIP] {seq_name}: vide.")
            continue

        min_len = window_size + output_size

        if len(seq_norm) < min_len:
            print(
                f"[SKIP] {seq_name}: trop courte "
                f"({len(seq_norm)}), min={min_len}"
            )
            continue

        # Limit the displayed length to window_size + output_size
        seq_norm_used = seq_norm[:min_len]

        # Complete x-axis starting from zero
        x_full = get_x_axis_full(
            len(seq_norm_used),
            use_minutes=USE_MINUTES,
            dt_min=DT_MIN
        )

        # Complete ground-truth curve starting from t=0
        gt_curve_full = mean_grain_size_from_distribution(
            seq_norm_used,
            BIN_CENTERS
        )

        pred_curves_full_by_model = {}

        for mt, model in models.items():

            preds_norm = autoregressive_predict_norm(
                model,
                seq_norm,
                window_size,
                output_size
            )

            pred_curve = mean_grain_size_from_distribution(
                preds_norm,
                BIN_CENTERS
            )

            # Complete curve containing NaN values before window_size
            pred_curve_full = np.full(len(gt_curve_full), np.nan, dtype=float)

            pred_curve_full[
                window_size:window_size + output_size
            ] = pred_curve

            pred_curves_full_by_model[mt] = pred_curve_full

            # Compute metrics only over the predicted part
            gt_for_metrics = gt_curve_full[
                window_size:window_size + output_size
            ]

            met = compute_metrics_1d(
                gt_for_metrics,
                pred_curve
            )

            global_metrics_by_model[mt].append(met)
            print_metrics(f"[{seq_name}][{mt.upper()}]", met)

        seq_out = os.path.join(
            out_root,
            "per_sequence",
            seq_name
        )

        save_sequence_mean_size_plot_and_csv(
            seq_name,
            x_full,
            gt_curve_full,
            pred_curves_full_by_model,
            seq_out
        )

        gt_curves_full_all.append(gt_curve_full)

        for mt in models.keys():
            if mt in pred_curves_full_by_model:
                pred_curves_full_all_by_model[mt].append(
                    pred_curves_full_by_model[mt]
                )

        gc.collect()

    if len(gt_curves_full_all) == 0:
        raise RuntimeError(
            "Aucune séquence valide pour calculer les courbes Mean Grain Size."
        )

    # ==========================
    # GLOBAL MEAN CURVES
    # ==========================
    gt_curves_full_all = np.stack(gt_curves_full_all, axis=0)
    gt_mean_full = np.nanmean(gt_curves_full_all, axis=0)

    mean_full_by_model = {}

    for mt, curves_list in pred_curves_full_all_by_model.items():
        if len(curves_list) == 0:
            continue

        arr = np.stack(curves_list, axis=0)
        mean_full_by_model[mt] = np.nanmean(arr, axis=0)

    x_full_global = get_x_axis_full(
        len(gt_mean_full),
        use_minutes=USE_MINUTES,
        dt_min=DT_MIN
    )

    global_out = os.path.join(out_root, "global")

    save_global_mean_plot_and_csv(
        x_full_global,
        gt_mean_full,
        mean_full_by_model,
        global_out
    )

    # ==========================
    # GLOBAL SUMMARY METRICS
    # ==========================
    summary_csv = os.path.join(
        global_out,
        "GLOBAL_METRICS_MEAN_GRAIN_SIZE_BY_MODEL.csv"
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
            if mt not in global_metrics_by_model:
                continue

            if len(global_metrics_by_model[mt]) == 0:
                continue

            mets = global_metrics_by_model[mt]

            w.writerow([
                mt,
                float(np.nanmean([m["MAE"] for m in mets])),
                float(np.nanmean([m["MSE"] for m in mets])),
                float(np.nanmean([m["RMSE"] for m in mets])),
                float(np.nanmean([m["MRE%"] for m in mets])),
                float(np.nanmean([m["R2"] for m in mets])),
            ])

    print("\n" + "=" * 80)
    print("[GLOBAL] Summary metrics saved:")
    print(f"  - {summary_csv}")
    print("=" * 80)

    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 80)
    print("FIN. Sorties générées:")
    print(
        f"  - {out_root}/per_sequence/<seq>/"
        f"mean_grain_size_over_time_ALL_MODELS.png + .csv"
    )
    print(
        f"  - {out_root}/global/"
        f"mean_grain_size_over_time_MEAN_ALL_MODELS.png + .csv"
    )
    print(
        f"  - {out_root}/global/"
        f"GLOBAL_METRICS_MEAN_GRAIN_SIZE_BY_MODEL.csv"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()