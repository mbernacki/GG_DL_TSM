# This script computes the MRE at each time step and saves the results in tables.
# It generates individual and combined MRE curves for all sequences and models.

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

# Directory containing the saved models and outputs
# base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/output_domain234"

# Directory containing the sequences to evaluate
# new_sequences_root = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/test_sequences3"

# base_output_dir = "/home/admin-eyounes/output"
# new_sequences_root = "/home/admin-eyounes/preprocessed_experimental_data"
base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/SN_output/1heure"

new_sequences_root = "/media/admin-eyounes/T7/1-Article/test_sequences"

# base_output_dir =  "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/output"
# new_sequences_root = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/test_sequences"
# base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/output/models"
# new_sequences_root ="/home/admin-eyounes/Desktop/29_janv/1/données+code2/3hours_4fev/Normale/test_sequences"

# Parameters must match the training configuration
datasize = 30
window_size = 5
output_size = 55
prediction_starts_at= 1 # The prediction starts at the second time step
AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types =["transformer", "rnn", "lstm", "tcn"]# Modify here if needed

# Output directory for MRE over time
mre_time_root = os.path.join(base_output_dir, "SN_mre_over_time_new_sequences_3heures")

# Fixed colors, identical to those used in the previous code
colors = {
    "y_test": "#444444",
    "rnn": "#1f7a1f",
    "lstm": "red",
    "tcn": "orange",
    "transformer": "blue",
}

# ==========================
# UTILITY FUNCTIONS
# ==========================
def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def numerical_sort(value):
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p for p in parts]

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


# ==========================
# NEW-SEQUENCE DATA LOADING
# ==========================
def load_one_sequence_in_memory(sequence_dir: str):
    """
    Loads one sequence directory into memory and returns:
      seq_norm: array with shape (T, datasize), normalized by its sum
      sums: array with shape (T,), containing the actual sums
      files: sorted list of .txt files
    """
    seq = []
    sums = []
    files = []

    for file_name in sorted(os.listdir(sequence_dir), key=numerical_sort):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(sequence_dir, file_name)
        with open(file_path, "r") as f:
            next(f)  # Skip the header
            vals = f.read().strip().split()
            vals = [float(v) for v in vals]

        if len(vals) != datasize:
            raise ValueError(f"Fichier {file_path}: attendu {datasize} valeurs, reçu {len(vals)}")

        s = sum(vals)
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
# AUTOREGRESSIVE PREDICTION WITH WINDOW RENORMALIZATION
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    seq_norm: normalized array with shape (T, datasize)
    Returns preds_norm with shape (output_size, datasize), containing the raw model outputs.
    The prediction is renormalized before it is used as the next input.
    """
    current_window = seq_norm[:window_size].copy()
    preds = []

    for _ in range(output_size):
        inp = current_window[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0]  # Shape: (datasize,)
        preds.append(pred)

        pred_sum = np.sum(pred) + 1e-8
        pred_norm_for_next = pred / pred_sum
        current_window = np.vstack((current_window[1:], pred_norm_for_next))

    return np.array(preds, dtype=float)


def predict_and_denormalize_new_sequences(model, new_sequences_root, window_size, output_size):
    """
    Iterates through the subdirectories of new_sequences_root.

    Returns:
      preds_norm: array with shape (N, output_size, datasize)
      y_norm: array with shape (N, output_size, datasize)
      seq_names: list
      file_names: dictionary {seq_name: [files...]}

    Important:
      No denormalization is performed here.
      The metrics are computed directly from the normalized distributions.
    """
    preds_list = []
    y_list = []
    seq_names = []
    file_names_map = {}

    if not os.path.exists(new_sequences_root):
        return None, None, [], {}

    for seq_name in sorted(os.listdir(new_sequences_root), key=numerical_sort):
        seq_path = os.path.join(new_sequences_root, seq_name)
        if not os.path.isdir(seq_path):
            continue

        seq_norm, sums, files = load_one_sequence_in_memory(seq_path)
        if seq_norm is None:
            continue

        file_names_map[seq_name] = files

        if len(seq_norm) < window_size + output_size:
            print(f"[NEW] {seq_name} trop courte ({len(seq_norm)}) -> ignorée. "
                  f"(min requis: {window_size + output_size})")
            continue

        preds_norm = autoregressive_predict_norm(model, seq_norm, window_size, output_size)

        # No denormalization:
        # Compare the normalized predictions directly with the normalized ground truth.
        preds_norm = np.maximum(preds_norm, 0)
        y_norm = seq_norm[window_size:window_size + output_size]

        preds_list.append(preds_norm)
        y_list.append(y_norm)
        seq_names.append(seq_name)

    if len(seq_names) == 0:
        return None, None, [], file_names_map

    return np.array(preds_list), np.array(y_list), seq_names, file_names_map


# ==========================
# ERROR REPORTING
# ==========================
def print_errors_global_and_per_sequence(denorm_preds, denorm_y, seq_names, label, model_type):
    print("\n" + "="*80)
    print(f"[ERREURS] {label} — {model_type.upper()}")
    print("="*80)

    if denorm_preds is None or len(denorm_preds) == 0:
        print("Aucune prédiction disponible.")
        return

    diff = denorm_preds - denorm_y
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    mask = denorm_y != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((denorm_y[mask] - denorm_preds[mask]) / denorm_y[mask])) * 100)
    else:
        mre = 0.0

    r2 = float(r2_score(denorm_y.flatten(), denorm_preds.flatten()))

    print(f"\nGlobal:")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MRE : {mre:.2f}%")
    print(f"  R²  : {r2:.4f}")

    print(f"\nPar séquence:")
    mae_list, mse_list, rmse_list, mre_list, r2_list = [], [], [], [], []

    for i, name in enumerate(seq_names):
        yt = denorm_y[i]
        yp = denorm_preds[i]
        d = yp - yt

        mae_i = float(np.mean(np.abs(d)))
        mse_i = float(np.mean(d**2))
        rmse_i = float(np.sqrt(mse_i))

        mask_i = (yt > 0) & (yp >= 0)
        if np.any(mask_i):
            rel = np.abs((yt[mask_i] - yp[mask_i]) / yt[mask_i])
            mre_i = float(np.sum(rel) / yt.size * 100)
        else:
            mre_i = 0.0

        r2_i = float(r2_score(yt.flatten(), yp.flatten()))

        mae_list.append(mae_i)
        mse_list.append(mse_i)
        rmse_list.append(rmse_i)
        mre_list.append(mre_i)
        r2_list.append(r2_i)

        print(f"  - {name}: MAE={mae_i:.4f} | MSE={mse_i:.4f} | RMSE={rmse_i:.4f} "
              f"| MRE={mre_i:.2f}% | R²={r2_i:.4f}")

    print(f"\nMoyenne sur séquences:")
    print(f"  mean MAE : {float(np.mean(mae_list)):.4f}")
    print(f"  mean MSE : {float(np.mean(mae_list)):.4f}")
    print(f"  mean RMSE: {float(np.mean(rmse_list)):.4f}")
    print(f"  mean MRE : {float(np.mean(mre_list)):.2f}%")
    print(f"  mean R²  : {float(np.mean(r2_list)):.4f}")


# ============================================================
# MRE OVER TIME FOR EACH TIME STEP
# ============================================================
def mre_percent_one_timestep(y_true_vec, y_pred_vec):
    """
    y_true_vec and y_pred_vec have shape (datasize,).
    MRE% is the sum of |(y-p)/y| where y>0, divided by datasize, and multiplied by 100.
    """
    y_true_vec = np.asarray(y_true_vec)
    y_pred_vec = np.asarray(y_pred_vec)

    mask = (y_true_vec > 0) & (y_pred_vec >= 0)
    if not np.any(mask):
        return 0.0

    rel = np.abs((y_true_vec[mask] - y_pred_vec[mask]) / y_true_vec[mask])
    return float(np.sum(rel) / y_true_vec.size * 100.0)


def save_mre_over_time_for_model(denorm_preds, denorm_y, seq_names,
                                 window_size, out_dir, model_type):
    """
    For each model:
      - saves one CSV and one PNG for each sequence
      - saves the mean CSV and PNG
      - saves the global summary CSV

    It also returns the curves for each sequence to generate
    the all-model comparison for every sequence.
    """
    safe_makedirs(out_dir)

    n_seq = denorm_preds.shape[0]
    T = denorm_preds.shape[1]

    all_curves = []
    all_global_vals = []

    per_seq_curve_map = {}
    per_seq_global_map = {}

    for i, seq_name in enumerate(seq_names):
        yt = denorm_y[i]
        yp = denorm_preds[i]

        curve = np.zeros((T,), dtype=float)
        for t in range(T):
            curve[t] = mre_percent_one_timestep(yt[t], yp[t])

        mask = (yt > 0) & (yp >= 0)
        if np.any(mask):
            rel = np.abs((yt[mask] - yp[mask]) / yt[mask])
            mre_global = float(np.sum(rel) / yt.size * 100.0)
        else:
            mre_global = 0.0

        per_seq_curve_map[seq_name] = curve
        per_seq_global_map[seq_name] = mre_global

        all_curves.append(curve)
        all_global_vals.append(mre_global)

        seq_out = os.path.join(out_dir, seq_name)
        safe_makedirs(seq_out)

        csv_path = os.path.join(seq_out, f"mre_over_time_{model_type}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_step_absolute", "time_step_horizon", "mre_percent"])
            for t in range(T):
                w.writerow([window_size + 1 + t, 1 + t, float(curve[t])])

        xs = np.arange(window_size + prediction_starts_at, window_size + prediction_starts_at + T)
        # xs = np.arange(window_size + 1, window_size + 1 + T)
        # plt.figure(figsize=(12, 8))
        plt.figure(figsize=(8, 6))
        plt.plot(xs, curve, marker="o", color=colors.get(model_type, "black"))
        plt.xlabel("Time step")
        plt.ylabel("MRE (%)")
        plt.title(f"")
        # plt.title(f"Mean MRE over time")
        # plt.title(f"{seq_name} - MRE over time ({model_type.upper()}) | global={mre_global:.2f}%")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(seq_out, f"mre_over_time_{model_type}.png")
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"[{model_type.upper()}][{seq_name}] saved: {csv_path} + {fig_path}")

    all_curves = np.stack(all_curves, axis=0)
    mean_curve = np.mean(all_curves, axis=0)
    mean_global_scalar = float(np.mean(all_global_vals)) if len(all_global_vals) > 0 else 0.0

    mean_csv = os.path.join(out_dir, f"mre_over_time_MEAN_{model_type}.csv")
    with open(mean_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_step_absolute", "time_step_horizon,mean_mre_percent"])
        for t in range(T):
            w.writerow([window_size + 1 + t, 1 + t, float(mean_curve[t])])

    # xs = np.arange(window_size + 1, window_size + 1 + T)
    xs = np.arange(window_size + prediction_starts_at, window_size + prediction_starts_at + T)
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 6))
    plt.plot(xs, mean_curve, marker="o", color=colors.get(model_type, "black"))
    plt.xlabel("Time step")
    plt.ylabel("Mean MRE (%)")
    plt.title(f"")
    # plt.title(f"Mean MRE over time ({model_type.upper()}) - {n_seq} sequences | global_mean={mean_global_scalar:.2f}%")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    mean_fig = os.path.join(out_dir, f"mre_over_time_MEAN_{model_type}.png")
    plt.savefig(mean_fig, dpi=300, bbox_inches="tight")
    plt.close()

    summary_csv = os.path.join(out_dir, f"mre_over_time_GLOBAL_{model_type}.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_name", "global_mre_percent"])
        for seq_name, g in per_seq_global_map.items():
            w.writerow([seq_name, float(g)])
        w.writerow(["MEAN_GLOBAL", mean_global_scalar])

    print(f"[{model_type.upper()}][MEAN] saved: {mean_csv} + {mean_fig}")
    print(f"[{model_type.upper()}][GLOBAL] summary saved: {summary_csv}")

    return mean_curve, mean_global_scalar, per_seq_curve_map, per_seq_global_map


def plot_combined_mean_mre_over_time(mean_curves_by_model, window_size, save_path, title):
    """
    Generates one plot comparing the mean MRE(t) curves of all models.
    """
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 6))
    for model_type, mean_curve in mean_curves_by_model.items():
        T = len(mean_curve)
        xs = np.arange(window_size + prediction_starts_at, window_size + prediction_starts_at + T)
        # xs = np.arange(window_size + 1, window_size + 1 + T)
        plt.plot(xs, mean_curve, marker="o", label=model_type.upper(), color=colors.get(model_type, None))

    plt.xlabel("Time step")
    plt.ylabel("Mean MRE (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ALL MODELS] combined MEAN saved: {save_path}")


def save_all_models_mre_over_time_per_sequence(per_seq_curve_by_model,
                                               per_seq_global_by_model,
                                               seq_names,
                                               window_size,
                                               out_root,
                                               model_types):
    """
    For each sequence:
      - saves one multi-model CSV file
      - saves one multi-model PNG containing four curves

    Output directory: out_root/ALL/<SEQ>/
    """
    all_dir = os.path.join(out_root, "ALL")
    safe_makedirs(all_dir)

    common = set(seq_names)
    for mt in model_types:
        if mt in per_seq_curve_by_model:
            common &= set(per_seq_curve_by_model[mt].keys())
    common = sorted(list(common), key=numerical_sort)

    if len(common) == 0:
        print("[ALL/SEQUENCES] aucune séquence commune à tous les modèles -> pas de plot ALL.")
        return

    first_mt = model_types[0]
    T = len(per_seq_curve_by_model[first_mt][common[0]])
    
    
    xs = np.arange(window_size + prediction_starts_at, window_size + prediction_starts_at + T)

    # xs = np.arange(window_size + 1, window_size + 1 + T)

    for seq_name in common:
        seq_out = os.path.join(all_dir, seq_name)
        safe_makedirs(seq_out)

        csv_path = os.path.join(seq_out, "mre_over_time_ALL_MODELS.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_step"] + [mt.upper() for mt in model_types])
            for i in range(T):
                row = [int(xs[i])]
                for mt in model_types:
                    row.append(float(per_seq_curve_by_model[mt][seq_name][i]))
                w.writerow(row)

        # plt.figure(figsize=(12, 8))
        plt.figure(figsize=(8, 6))
        for mt in model_types:
            curve = per_seq_curve_by_model[mt][seq_name]
            g = per_seq_global_by_model[mt][seq_name]
            plt.plot(xs, curve, marker="o",
                     label=f"{mt.upper()} (global={g:.2f}%)",
                     color=colors.get(mt, None))

        plt.xlabel("Time step")
        plt.ylabel("MRE (%)")
        plt.title(f"")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="best", framealpha=0.9)
        plt.tight_layout()

        fig_path = os.path.join(seq_out, "mre_over_time_ALL_MODELS.png")
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"[ALL/{seq_name}] saved: {csv_path} + {fig_path}")


# ==========================
# MAIN
# ==========================
def main():
    print("="*80)
    print("NEW SEQUENCES ONLY (LOAD SAVED MODEL .keras)")
    print("="*80)

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}")

    if not os.path.exists(new_sequences_root):
        raise FileNotFoundError(f"Dossier new_sequences_root introuvable: {new_sequences_root}")

    reset_directory(mre_time_root)

    mean_curves_by_model = {}
    mean_global_by_model = {}

    # Store per-sequence curves for all-model comparisons
    per_seq_curve_by_model = {}
    per_seq_global_by_model = {}

    seq_names_ref = None

    for model_type in model_types:
        print("\n" + "="*80)
        print(f"[LOAD MODEL] {model_type.upper()}")
        print("="*80)

        model_path = os.path.join(base_output_dir, f"model_{model_type}.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")

        model = tf.keras.models.load_model(model_path)
        print(f"  ✅ Modèle chargé: {model_path}")

        print(f"\n[NEW] Prédictions SANS dénormalisation ({model_type.upper()}) ...")
        preds_norm_new, y_norm_new, new_names, _ = predict_and_denormalize_new_sequences(
            model, new_sequences_root, window_size, output_size
        )

        if preds_norm_new is None or len(new_names) == 0:
            print("[NEW] Aucune nouvelle séquence valide (vide ou trop courte).")
            tf.keras.backend.clear_session()
            continue

        if seq_names_ref is None:
            seq_names_ref = new_names
        else:
            if new_names != seq_names_ref:
                print("[WARN] ordre des nouvelles séquences différent selon modèle.")

        print_errors_global_and_per_sequence(
            preds_norm_new, y_norm_new, new_names,
            label="NEW SEQUENCES", model_type=model_type
        )

        model_out_dir = os.path.join(mre_time_root, model_type)
        mean_curve, mean_global_scalar, curve_map, global_map = save_mre_over_time_for_model(
            denorm_preds=preds_norm_new,
            denorm_y=y_norm_new,
            seq_names=new_names,
            window_size=window_size,
            out_dir=model_out_dir,
            model_type=model_type
        )

        mean_curves_by_model[model_type] = mean_curve
        mean_global_by_model[model_type] = mean_global_scalar
        per_seq_curve_by_model[model_type] = curve_map
        per_seq_global_by_model[model_type] = global_map

        tf.keras.backend.clear_session()
        gc.collect()

    # Generate the combined plot comparing the mean model curves
    if len(mean_curves_by_model) > 0:
        combined_path = os.path.join(mre_time_root, "mre_over_time_MEAN_ALL_MODELS.png")
        plot_combined_mean_mre_over_time(
            mean_curves_by_model=mean_curves_by_model,
            window_size=window_size,
            save_path=combined_path,
            title=""
        )

        summary_path = os.path.join(mre_time_root, "GLOBAL_MEAN_MRE_BY_MODEL.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_type", "mean_global_mre_percent"])
            for m in model_types:
                if m in mean_global_by_model:
                    w.writerow([m, float(mean_global_by_model[m])])
        print(f"[ALL MODELS] global mean summary saved: {summary_path}")

    # Generate one four-curve comparison plot for every sequence
    if seq_names_ref is not None and len(per_seq_curve_by_model) > 0:
        save_all_models_mre_over_time_per_sequence(
            per_seq_curve_by_model=per_seq_curve_by_model,
            per_seq_global_by_model=per_seq_global_by_model,
            seq_names=seq_names_ref,
            window_size=window_size,
            out_root=mre_time_root,
            model_types=model_types
        )

    print("\n" + "="*80)
    print("FIN. Sorties générées:")
    print(f"  - {mre_time_root}/")
    print("    -> par modèle / par séquence: <model>/<seq>/mre_over_time_<model>.csv + .png")
    print("    -> par modèle: <model>/mre_over_time_MEAN_<model>.csv + .png")
    print("    -> ALL models mean: mre_over_time_MEAN_ALL_MODELS.png")
    print("    -> ALL models per sequence: ALL/<seq>/mre_over_time_ALL_MODELS.png + .csv")
    print("="*80)


if __name__ == "__main__":
    main()