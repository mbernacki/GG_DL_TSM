# This script generates normalized predictions for new sequences using trained models.
# The final bins are excluded from the figures, which are generated at selected time steps.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import gc

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl

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
xmax = 0.17

# Number of bins excluded from the right side of the figures only
IGNORED_BINS_IN_PLOT = 5

# Generate one figure every four time steps:
# Time Step 6, 10, 14, 18, ...
FIGURE_STEP = 2

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/SN_output/3heures"
new_sequences_root = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/test_sequences"

datasize = 30
window_size = 5
output_size = 175

AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types = ["rnn", "lstm", "tcn", "transformer"]

figures_new_root = os.path.join(
    base_output_dir,
    "4SN_figures_selected_new_sequences_3heures_NORMALIZED"
)

# ==========================
# UTILITY FUNCTIONS
# ==========================
def reset_directory(directory: str):
    os.makedirs(directory, exist_ok=True)

def numerical_sort(value):
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p for p in parts]

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

# ==========================
# DATA LOADING
# ==========================
def load_one_sequence_in_memory(sequence_dir: str):
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
# PREDICTION
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    current_window = seq_norm[:window_size].copy()
    preds = []

    for _ in range(output_size):
        inp = current_window[np.newaxis, ...]
        pred = model.predict(inp, verbose=0)[0]
        preds.append(pred)

        pred_sum = np.sum(pred) + 1e-8
        pred_norm_for_next = pred / pred_sum

        current_window = np.vstack((current_window[1:], pred_norm_for_next))

    return np.array(preds, dtype=float)

def predict_new_sequences_norm(model, new_sequences_root, window_size, output_size):
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
            print(
                f"[NEW] {seq_name} trop courte ({len(seq_norm)}) -> ignorée. "
                f"(min requis: {window_size + output_size})"
            )
            continue

        preds_norm = autoregressive_predict_norm(
            model,
            seq_norm,
            window_size,
            output_size
        )
        
        norm_preds = preds_norm
        # norm_preds = np.maximum(preds_norm, 0)
        norm_y = seq_norm[window_size:window_size + output_size]

        preds_list.append(norm_preds)
        y_list.append(norm_y)
        seq_names.append(seq_name)

    if len(seq_names) == 0:
        return None, None, [], file_names_map

    return np.array(preds_list), np.array(y_list), seq_names, file_names_map

# ==========================
# ERRORS
# ==========================
def print_errors_global_and_per_sequence(norm_preds, norm_y, seq_names, label, model_type):
    print("\n" + "=" * 80)
    print(f"[ERREURS] {label} — {model_type.upper()}")
    print("=" * 80)

    if norm_preds is None or len(norm_preds) == 0:
        print("Aucune prédiction disponible.")
        return

    diff = norm_preds - norm_y

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    mask = norm_y != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((norm_y[mask] - norm_preds[mask]) / norm_y[mask])) * 100)
    else:
        mre = 0.0

    r2 = float(r2_score(norm_y.flatten(), norm_preds.flatten()))

    print("\nGlobal:")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MRE : {mre:.2f}%")
    print(f"  R²  : {r2:.4f}")

    print("\nPar séquence:")

    mae_list = []
    mse_list = []
    rmse_list = []
    mre_list = []
    r2_list = []

    for i, name in enumerate(seq_names):
        yt = norm_y[i]
        yp = norm_preds[i]
        d = yp - yt

        mae_i = float(np.mean(np.abs(d)))
        mse_i = float(np.mean(d ** 2))
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

        print(
            f"  - {name}: MAE={mae_i:.4f} | MSE={mse_i:.4f} | "
            f"RMSE={rmse_i:.4f} | MRE={mre_i:.2f}% | R²={r2_i:.4f}"
        )

    print("\nMoyenne sur séquences:")
    print(f"  mean MAE : {float(np.mean(mae_list)):.4f}")
    print(f"  mean MSE : {float(np.mean(mse_list)):.4f}")
    print(f"  mean RMSE: {float(np.mean(rmse_list)):.4f}")
    print(f"  mean MRE : {float(np.mean(mre_list)):.2f}%")
    print(f"  mean R²  : {float(np.mean(r2_list)):.4f}")

# ==========================
# FIGURES
# ==========================
def generate_combined_figures_generic(
    all_predictions,
    all_y_test,
    seq_names,
    output_size,
    datasize,
    out_root,
    model_types,
    window_size,
    xmax,
    yscale_mode="per_timestep",
    fixed_ylim=None,
    ignored_bins_in_plot=IGNORED_BINS_IN_PLOT,
    figure_step=FIGURE_STEP
):
    safe_makedirs(out_root)

    if ignored_bins_in_plot < 0 or ignored_bins_in_plot >= datasize:
        raise ValueError("ignored_bins_in_plot doit être >= 0 et < datasize")

    x_min = 0.0
    bin_width = xmax / datasize

    effective_bins = datasize - ignored_bins_in_plot
    x_max_eff = xmax - ignored_bins_in_plot * bin_width

    bin_edges = np.linspace(x_min, x_max_eff, effective_bins + 1)
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    x_groups = x_centers

    total_bars = len(model_types) + 1

    group_fill = 0.75
    bar_width = bin_width * group_fill / total_bars
    edge_pad = bin_width * (1 - group_fill) / 2

    colors = {
        "y_test": "#444444",
        "rnn": "#1f7a1f",
        "lstm": "red",
        "tcn": "orange",
        "transformer": "blue",
    }

    custom_labels = {
        "y_test": "Ground truth",
        "rnn": "RNN prediction",
        "lstm": "LSTM prediction",
        "tcn": "TCN prediction",
        "transformer": "TRANSFORMER prediction",
    }

    first_model = model_types[0]

    for seq_idx, seq_name in enumerate(seq_names):
        seq_dir = os.path.join(out_root, seq_name)
        safe_makedirs(seq_dir)

        # Generate only the selected time steps
        # t = 0  -> Time Step 6
        # t = 4  -> Time Step 10
        # t = 8  -> Time Step 14
        # etc.
        for t in range(0, output_size, figure_step):

            ts_max = float(np.max(all_y_test[first_model][seq_idx, t, :effective_bins]))

            for mt in model_types:
                ts_max = max(
                    ts_max,
                    float(np.max(all_predictions[mt][seq_idx, t, :effective_bins]))
                )

            y_limit = ts_max * 1.1 if ts_max > 0 else 1.0

            plt.figure(figsize=(8, 6))

            y_true_vec = all_y_test[first_model][seq_idx, t, :]

            max_idx = 0

            for key in ["y_test"] + model_types:
                if key == "y_test":
                    vec_full = y_true_vec
                else:
                    vec_full = all_predictions[key][seq_idx, t, :]

                vec = vec_full[:effective_bins]
                nz = np.where(vec > 0)[0]

                if len(nz) > 0:
                    max_idx = max(max_idx, int(nz.max()))

            right_idx = min(max_idx + 1, effective_bins - 1)
            x_right = x_groups[right_idx] + bin_width

            plt.xlim(x_min - edge_pad, x_right)

            for bin_idx, x_val in enumerate(x_groups):
                for model_offset, key in enumerate(["y_test"] + model_types):

                    if key == "y_test":
                        y = all_y_test[first_model][seq_idx, t, bin_idx]
                    else:
                        y = all_predictions[key][seq_idx, t, bin_idx]

                    offset = (model_offset - (total_bars - 1) / 2.0) * bar_width
                    x_pos = x_val + offset

                    if y > 0:
                        plt.bar(
                            x_pos,
                            y,
                            width=bar_width,
                            color=colors.get(key, "gray"),
                            label=custom_labels.get(key, key.upper()) if bin_idx == 0 else "",
                            edgecolor="white",
                            linewidth=0.8,
                            alpha=0.9
                        )
                    else:
                        plt.scatter(
                            x_pos,
                            0,
                            color=colors.get(key, "gray"),
                            s=18,
                            zorder=5
                        )

            plt.xlim(x_min - edge_pad, x_max_eff + edge_pad)
            plt.margins(x=0)
            plt.ylim(0, y_limit)

            true_time_step = t + window_size + 1

            plt.xlabel("Grain Size (mm)", fontsize=19)
            plt.ylabel("Normalized frequency", fontsize=19)
            plt.title(f"Time Step {true_time_step}")

            plt.xticks(
                x_groups[::2],
                [f"{x:.3f}" for x in x_groups[::2]],
                rotation=45
            )

            plt.grid(True, linestyle="--", alpha=0.3)

            plt.legend(
                loc="upper right",
                fontsize=9,
                ncol=1,
                borderpad=0.3,
                labelspacing=0.3,
                columnspacing=0.8,
                handlelength=1.2,
                borderaxespad=0.4
            )

            plt.tight_layout()

            # Use a clear filename containing the actual time step
            fig_path = os.path.join(seq_dir, f"time_step_{true_time_step}.png")

            plt.savefig(fig_path)
            plt.close()
            gc.collect()

# ==========================
# MAIN
# ==========================
def main():
    print("=" * 80)
    print("NEW SEQUENCES ONLY (LOAD SAVED MODEL .keras)")
    print("=" * 80)

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}")

    if not os.path.exists(new_sequences_root):
        raise FileNotFoundError(
            f"Dossier new_sequences_root introuvable: {new_sequences_root}"
        )

    reset_directory(figures_new_root)

    all_predictions_new = {}
    all_y_new = {}
    new_seq_names_ref = None

    for model_type in model_types:
        print("\n" + "=" * 80)
        print(f"[LOAD MODEL] {model_type.upper()}")
        print("=" * 80)

        model_path = os.path.join(base_output_dir, f"model_{model_type}.keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")

        model = tf.keras.models.load_model(model_path)

        print(f"  ✅ Modèle chargé: {model_path}")

        print(
            f"\n[NEW] Prédictions normalisées SANS dénormalisation "
            f"({model_type.upper()}) ..."
        )

        norm_preds_new, norm_y_new, new_names, _ = predict_new_sequences_norm(
            model,
            new_sequences_root,
            window_size,
            output_size
        )

        if norm_preds_new is None or len(new_names) == 0:
            print("[NEW] Aucune nouvelle séquence valide.")
            tf.keras.backend.clear_session()
            continue

        all_predictions_new[model_type] = norm_preds_new
        all_y_new[model_type] = norm_y_new

        if new_seq_names_ref is None:
            new_seq_names_ref = new_names
        else:
            if new_names != new_seq_names_ref:
                print("[WARN] ordre des nouvelles séquences différent.")
                new_seq_names_ref = new_names

        print_errors_global_and_per_sequence(
            norm_preds_new,
            norm_y_new,
            new_names,
            label="NEW SEQUENCES",
            model_type=model_type
        )

        tf.keras.backend.clear_session()
        gc.collect()

    if (
        new_seq_names_ref is not None
        and len(new_seq_names_ref) > 0
        and len(all_predictions_new) > 0
    ):
        print("\n" + "=" * 80)
        print("[FIGURES] Génération des figures sélectionnées")
        print("          Time Step 6, 10, 14, 18, ...")
        print("=" * 80)

        generate_combined_figures_generic(
            all_predictions=all_predictions_new,
            all_y_test=all_y_new,
            seq_names=new_seq_names_ref,
            output_size=output_size,
            datasize=datasize,
            out_root=figures_new_root,
            model_types=model_types,
            window_size=window_size,
            xmax=xmax,
            yscale_mode="per_timestep",
            ignored_bins_in_plot=IGNORED_BINS_IN_PLOT,
            figure_step=FIGURE_STEP
        )

        print(f"  ✅ Figures sélectionnées dans: {figures_new_root}")

    else:
        print("\n[FIGURES] Aucune séquence valide -> pas de figures.")

    print("\n" + "=" * 80)
    print("FIN. Sorties générées:")
    print(f"  - {figures_new_root}")
    print("=" * 80)

if __name__ == "__main__":
    main()