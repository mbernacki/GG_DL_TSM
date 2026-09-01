# This script computes R² at each time step for normalized predictions from trained forecasting models.
# It saves per-sequence and global R² curves, comparison figures, and CSV summaries.

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


mpl.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 18,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.autolayout": True,
    "lines.linewidth": 3.5,
})

# mpl.rcParams.update({
#     "figure.dpi": 600,
#     "savefig.dpi": 600,
#     "font.size": 18,
#     "axes.titlesize": 15,
#     "axes.labelsize": 15,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 12,
#     "figure.autolayout": True,
#     "lines.linewidth": 2.5,
# })


# ==========================
# CONFIG (à adapter)
# ==========================
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Directory containing the sequences to evaluate
base_output_dir = "/media/admin-eyounes/T7/1-Article/github_code/SN_output/1heure"
new_sequences_root = "/media/admin-eyounes/T7/1-Article/github_code/test_sequences"


# Parameters must match the training configuration
datasize = 30
window_size = 5
output_size = 55

AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types = ["rnn", "lstm", "tcn", "transformer"]  # <- modifie ici si besoin

# Output directory for R² over time
r2_time_root = os.path.join(base_output_dir, "SN_r2_over_time_new_sequences_3heures")

# Fixed colors, identical to those used in the previous code
colors = {
    "y_test": "#444444",
    "rnn": "#1f7a1f",
    "lstm": "red",
    "tcn": "orange",
    "transformer": "blue",
}

# ==========================
# UTILS
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
# DATA LOADING (NEW SEQUENCES)
# ==========================
def load_one_sequence_in_memory(sequence_dir: str):
    """
    Charge une séquence (dossier) en mémoire, renvoie:
      seq_norm: (T, datasize) normalisé par somme
      sums:     (T,) sommes réelles
      files:    liste des fichiers .txt (ordre trié)
    """
    seq = []
    sums = []
    files = []

    for file_name in sorted(os.listdir(sequence_dir), key=numerical_sort):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(sequence_dir, file_name)
        with open(file_path, "r") as f:
            next(f)  # skip header
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
# PREDICTION (AUTO-REGRESSIVE + RENORMALIZE WINDOW)
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    seq_norm: (T, datasize) normalisé
    Retourne preds_norm: (output_size, datasize) (sorties brutes du modèle)
    Mais renormalise la prédiction pour l'entrée suivante.
    """
    current_window = seq_norm[:window_size].copy()
    preds = []

    for _ in range(output_size):
        inp = current_window[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0]  # (datasize,)
        preds.append(pred)

        pred_sum = np.sum(pred) + 1e-8
        pred_norm_for_next = pred / pred_sum
        current_window = np.vstack((current_window[1:], pred_norm_for_next))

    return np.array(preds, dtype=float)

def predict_normalized_new_sequences(model, new_sequences_root, window_size, output_size):
    """
    Parcourt new_sequences_root (sous-dossiers).
    Retourne:
      norm_preds: (N, output_size, datasize)
      norm_y    : (N, output_size, datasize)
      seq_names : liste
      file_names: dict {seq_name: [files...]}

    IMPORTANT:
      Ici on ne fait aucune dénormalisation.
      Les métriques sont calculées directement sur les distributions normalisées.
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

        # Sans dénormalisation:
        # on compare directement les prédictions normalisées
        # avec la vérité terrain normalisée.
        norm_preds = np.maximum(preds_norm, 0)
        norm_y = seq_norm[window_size:window_size + output_size]

        preds_list.append(norm_preds)
        y_list.append(norm_y)
        seq_names.append(seq_name)

    if len(seq_names) == 0:
        return None, None, [], file_names_map

    return np.array(preds_list), np.array(y_list), seq_names, file_names_map

# ==========================
# ERRORS (PRINT)
# ==========================
def print_errors_global_and_per_sequence(norm_preds, norm_y, seq_names, label, model_type):
    print("\n" + "="*80)
    print(f"[ERREURS] {label} — {model_type.upper()}")
    print("="*80)

    if norm_preds is None or len(norm_preds) == 0:
        print("Aucune prédiction disponible.")
        return

    diff = norm_preds - norm_y
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    mask = norm_y != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((norm_y[mask] - norm_preds[mask]) / norm_y[mask])) * 100)
    else:
        mre = 0.0

    r2 = float(r2_score(norm_y.flatten(), norm_preds.flatten()))

    print(f"\nGlobal:")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MRE : {mre:.2f}%")
    print(f"  R²  : {r2:.4f}")

    print(f"\nPar séquence:")
    mae_list, mse_list, rmse_list, mre_list, r2_list = [], [], [], [], []

    for i, name in enumerate(seq_names):
        yt = norm_y[i]
        yp = norm_preds[i]
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
    print(f"  mean MSE : {float(np.mean(mse_list)):.4f}")
    print(f"  mean RMSE: {float(np.mean(rmse_list)):.4f}")
    print(f"  mean MRE : {float(np.mean(mre_list)):.2f}%")
    print(f"  mean R²  : {float(np.mean(r2_list)):.4f}")

# ============================================================
# R2 over time (per time step)
# ============================================================
def r2_one_timestep(y_true_vec, y_pred_vec):
    """
    R² calculé sur les 30 bins pour un time step.
    Retourne NaN si y_true_vec est constant (variance nulle).
    """
    y_true_vec = np.asarray(y_true_vec, dtype=float).flatten()
    y_pred_vec = np.asarray(y_pred_vec, dtype=float).flatten()

    if np.allclose(np.var(y_true_vec), 0.0):
        return float("nan")

    return float(r2_score(y_true_vec, y_pred_vec))

def save_r2_over_time_for_model(norm_preds, norm_y, seq_names,
                                window_size, out_dir, model_type):
    """
    Par modèle:
      - Sauve per-seq CSV+PNG (1 courbe)
      - Sauve mean CSV+PNG
      - Sauve global summary CSV
    ✅ Retourne aussi les courbes per séquence pour faire ALL MODELS par séquence.
    """
    safe_makedirs(out_dir)

    n_seq = norm_preds.shape[0]
    T = norm_preds.shape[1]

    all_curves = []
    all_global_vals = []

    per_seq_curve_map = {}
    per_seq_global_map = {}

    for i, seq_name in enumerate(seq_names):
        yt = norm_y[i]
        yp = norm_preds[i]

        curve = np.zeros((T,), dtype=float)
        for t in range(T):
            curve[t] = r2_one_timestep(yt[t], yp[t])

        r2_global = float(r2_score(yt.flatten(), yp.flatten()))

        per_seq_curve_map[seq_name] = curve
        per_seq_global_map[seq_name] = r2_global

        all_curves.append(curve)
        all_global_vals.append(r2_global)

        seq_out = os.path.join(out_dir, seq_name)
        safe_makedirs(seq_out)

        csv_path = os.path.join(seq_out, f"r2_over_time_{model_type}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_step_absolute", "time_step_horizon", "r2"])
            for t in range(T):
                w.writerow([window_size + 1 + t, 1 + t, float(curve[t])])

        xs = np.arange(window_size + 1, window_size + 1 + T)
        plt.figure(figsize=(8, 6))
        # plt.figure(figsize=(12, 8))
        plt.plot(xs, curve, marker="o", color=colors.get(model_type, "black"))
        plt.xlabel("Time step")
        plt.ylabel("R²")
        plt.title(f"")
        # plt.title(f"{seq_name} - R² over time ({model_type.upper()}) | global={r2_global:.4f}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(seq_out, f"r2_over_time_{model_type}.png")
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"[{model_type.upper()}][{seq_name}] saved: {csv_path} + {fig_path}")

    all_curves = np.stack(all_curves, axis=0)
    mean_curve = np.nanmean(all_curves, axis=0)
    mean_global_scalar = float(np.mean(all_global_vals)) if len(all_global_vals) > 0 else float("nan")

    mean_csv = os.path.join(out_dir, f"r2_over_time_MEAN_{model_type}.csv")
    with open(mean_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_step_absolute", "time_step_horizon", "mean_r2"])
        for t in range(T):
            w.writerow([window_size + 1 + t, 1 + t, float(mean_curve[t])])

    xs = np.arange(window_size + 1, window_size + 1 + T)
    plt.figure(figsize=(8, 6))
    #plt.figure(figsize=(12, 8))
    plt.plot(xs, mean_curve, marker="o", color=colors.get(model_type, "black"))
    plt.xlabel("Time step")
    plt.ylabel("Mean R²")
    plt.title(f"")
    # plt.title(f"Mean R² over time ({model_type.upper()}) - {n_seq} sequences | global_mean={mean_global_scalar:.4f}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    mean_fig = os.path.join(out_dir, f"r2_over_time_MEAN_{model_type}.png")
    plt.savefig(mean_fig, dpi=600, bbox_inches="tight")
    plt.close()

    summary_csv = os.path.join(out_dir, f"r2_over_time_GLOBAL_{model_type}.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_name", "global_r2"])
        for seq_name, g in per_seq_global_map.items():
            w.writerow([seq_name, float(g)])
        w.writerow(["MEAN_GLOBAL", float(mean_global_scalar)])

    print(f"[{model_type.upper()}][MEAN] saved: {mean_csv} + {mean_fig}")
    print(f"[{model_type.upper()}][GLOBAL] summary saved: {summary_csv}")

    return mean_curve, mean_global_scalar, per_seq_curve_map, per_seq_global_map

def plot_combined_mean_r2_over_time(mean_curves_by_model, window_size, save_path, title):
    """
    One plot comparing models: mean R²(t) curves.
    """
    plt.figure(figsize=(8, 6))
    #plt.figure(figsize=(12, 8))
    for model_type, mean_curve in mean_curves_by_model.items():
        T = len(mean_curve)
        xs = np.arange(window_size + 1, window_size + 1 + T)
        plt.plot(xs, mean_curve, marker="o", label=model_type.upper(), color=colors.get(model_type, None))

    plt.xlabel("Time step")
    plt.ylabel("Mean R²")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ALL MODELS] combined MEAN saved: {save_path}")

def save_all_models_r2_over_time_per_sequence(per_seq_curve_by_model,
                                              per_seq_global_by_model,
                                              seq_names,
                                              window_size,
                                              out_root,
                                              model_types):
    """
    Pour chaque séquence:
      - 1 CSV multi-modèles
      - 1 PNG multi-modèles (4 courbes)
    Sortie: out_root/ALL/<SEQ>/
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
    xs = np.arange(window_size + 1, window_size + 1 + T)

    for seq_name in common:
        seq_out = os.path.join(all_dir, seq_name)
        safe_makedirs(seq_out)

        csv_path = os.path.join(seq_out, "r2_over_time_ALL_MODELS.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_step"] + [mt.upper() for mt in model_types])
            for i in range(T):
                row = [int(xs[i])]
                for mt in model_types:
                    row.append(float(per_seq_curve_by_model[mt][seq_name][i]))
                w.writerow(row)

        plt.figure(figsize=(8, 6))
        for mt in model_types:
            curve = per_seq_curve_by_model[mt][seq_name]
            g = per_seq_global_by_model[mt][seq_name]
            plt.plot(xs, curve, marker="o",
                     label=f"{mt.upper()} (global={g:.4f})",
                     color=colors.get(mt, None))

        plt.xlabel("Time step")
        plt.ylabel("R²")
        plt.title(f"")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="best", framealpha=0.9)
        plt.tight_layout()

        fig_path = os.path.join(seq_out, "r2_over_time_ALL_MODELS.png")
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"[ALL/{seq_name}] saved: {csv_path} + {fig_path}")

# ==========================
# MAIN
# ==========================
def main():
    print("="*80)
    print("NEW SEQUENCES ONLY (LOAD SAVED MODEL .keras) — R2 OVER TIME")
    print("="*80)

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}")

    if not os.path.exists(new_sequences_root):
        raise FileNotFoundError(f"Dossier new_sequences_root introuvable: {new_sequences_root}")

    reset_directory(r2_time_root)

    mean_curves_by_model = {}
    mean_global_by_model = {}

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

        print(f"\n[NEW] Prédictions sans dénormalisation ({model_type.upper()}) ...")
        norm_preds_new, norm_y_new, new_names, _ = predict_normalized_new_sequences(
            model, new_sequences_root, window_size, output_size
        )

        if norm_preds_new is None or len(new_names) == 0:
            print("[NEW] Aucune nouvelle séquence valide (vide ou trop courte).")
            tf.keras.backend.clear_session()
            continue

        if seq_names_ref is None:
            seq_names_ref = new_names
        else:
            if new_names != seq_names_ref:
                print("[WARN] ordre des nouvelles séquences différent selon modèle.")

        print_errors_global_and_per_sequence(
            norm_preds_new, norm_y_new, new_names,
            label="NEW SEQUENCES", model_type=model_type
        )

        model_out_dir = os.path.join(r2_time_root, model_type)
        mean_curve, mean_global_scalar, curve_map, global_map = save_r2_over_time_for_model(
            norm_preds=norm_preds_new,
            norm_y=norm_y_new,
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

    # combined plot comparing models (mean curves)
    if len(mean_curves_by_model) > 0:
        combined_path = os.path.join(r2_time_root, "r2_over_time_MEAN_ALL_MODELS.png")
        plot_combined_mean_r2_over_time(
            mean_curves_by_model=mean_curves_by_model,
            window_size=window_size,
            save_path=combined_path,
            title=""
        )

        summary_path = os.path.join(r2_time_root, "GLOBAL_MEAN_R2_BY_MODEL.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_type", "mean_global_r2"])
            for m in model_types:
                if m in mean_global_by_model:
                    w.writerow([m, float(mean_global_by_model[m])])
        print(f"[ALL MODELS] global mean summary saved: {summary_path}")

    # pour chaque séquence -> 1 plot avec 4 courbes
    if seq_names_ref is not None and len(per_seq_curve_by_model) > 0:
        save_all_models_r2_over_time_per_sequence(
            per_seq_curve_by_model=per_seq_curve_by_model,
            per_seq_global_by_model=per_seq_global_by_model,
            seq_names=seq_names_ref,
            window_size=window_size,
            out_root=r2_time_root,
            model_types=model_types
        )

    print("\n" + "="*80)
    print("FIN. Sorties générées:")
    print(f"  - {r2_time_root}/")
    print("    -> par modèle / par séquence: <model>/<seq>/r2_over_time_<model>.csv + .png")
    print("    -> par modèle: <model>/r2_over_time_MEAN_<model>.csv + .png")
    print("    -> ALL models mean: r2_over_time_MEAN_ALL_MODELS.png")
    print("    -> ALL models per sequence: ALL/<seq>/r2_over_time_ALL_MODELS.png + .csv")
    print("="*80)

if __name__ == "__main__":
    main()