# This script trains the selected model, evaluates its normalized predictions, and saves the Keras model.
# It also contains the functions required to generate combined figures for test and new sequences.

# Suppress TensorFlow warnings before importing TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import shutil
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 300,
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
# CONFIG
# ==========================
xmax = 0.12

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
data_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/trm"
base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/output_10juillet/test"
new_sequences_root = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/test_sequences"

datasize = 30
window_size = 5
output_size = 55

AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types = ["transformer"]  # <- modifie ici

os.makedirs(base_output_dir, exist_ok=True)

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
# DATA LOADING (NO FILE OUTPUT)
# ==========================
def load_and_normalize_data_in_memory(data_dir: str):
    """
    Lit chaque sous-dossier de data_dir (une séquence),
    lit chaque .txt, calcule :
      - normalized_values = vals / sum(vals)
      - sums = sum(vals)
    Retourne tout en mémoire, NE SAUVEGARDE RIEN.
    """
    sequences = []
    sequences_sums = []
    file_names = []
    sequence_dirs = []

    for sequence_dir in sorted(os.listdir(data_dir), key=numerical_sort):
        sequence_path = os.path.join(data_dir, sequence_dir)
        if not os.path.isdir(sequence_path):
            continue

        seq = []
        seq_sums = []
        seq_files = []

        for file_name in sorted(os.listdir(sequence_path), key=numerical_sort):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(sequence_path, file_name)
            with open(file_path, "r") as f:
                next(f)  # skip header
                vals = f.read().strip().split()
                vals = [float(v) for v in vals]

            s = sum(vals)
            seq_sums.append(s)
            if s != 0:
                norm = [round(v / s, 16) for v in vals]
            else:
                norm = [0.0 for _ in vals]

            seq.append(norm)
            seq_files.append(file_name)

        if len(seq) == 0:
            continue

        sequences.append(seq)
        sequences_sums.append(seq_sums)
        file_names.append(seq_files)
        sequence_dirs.append(sequence_dir)

    sequences_array = np.array([np.array(seq) for seq in sequences], dtype=float)
    sequences_sums_array = np.array(sequences_sums, dtype=float)

    print("Shape of sequences_array:", sequences_array.shape)
    print("Shape of sequences_sums_array:", sequences_sums_array.shape)

    return sequences_array, sequences_sums_array, file_names, sequence_dirs

def load_one_sequence_in_memory(sequence_dir: str):
    """
    Charge une séquence (dossier) en mémoire, renvoie:
      sequence_norm: (T, datasize)
      sums: (T,)
      file_names: liste
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
# WINDOWS + SPLITS
# ==========================
def create_sliding_windows(sequence, sums, window_size):
    X, Y, Y_sums = [], [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        Y.append(sequence[i + window_size])
        Y_sums.append(sums[i + window_size])
    return np.array(X), np.array(Y), np.array(Y_sums)

def prepare_datasets(sequences_array, sequences_sums_array, sequence_dirs, window_size):
    # split by sequence  er (not by windows)
    train_idx, val_test_idx = train_test_split(
        range(len(sequence_dirs)), test_size=0.2, random_state=43
    )
    val_idx, test_idx = train_test_split(
        # val_test_idx, test_size=0.25, random_state=42
        val_test_idx, test_size=0.5, random_state=42
    )

    seq_dirs_train = [sequence_dirs[i] for i in train_idx]
    seq_dirs_validation = [sequence_dirs[i] for i in val_idx]
    seq_dirs_test = [sequence_dirs[i] for i in test_idx]

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for seq_idx in range(sequences_array.shape[0]):
        X_seq, Y_seq, _ = create_sliding_windows(
            sequences_array[seq_idx], sequences_sums_array[seq_idx], window_size
        )
        name = sequence_dirs[seq_idx]
        if name in seq_dirs_train:
            X_train.append(X_seq); Y_train.append(Y_seq)
        elif name in seq_dirs_validation:
            X_val.append(X_seq); Y_val.append(Y_seq)
        elif name in seq_dirs_test:
            X_test.append(X_seq); Y_test.append(Y_seq)

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val   = np.concatenate(X_val, axis=0)
    Y_val   = np.concatenate(Y_val, axis=0)
    X_test  = np.concatenate(X_test, axis=0)
    Y_test  = np.concatenate(Y_test, axis=0)

    print("Shape of X_train:", X_train.shape)
    print("Shape of Y_train:", Y_train.shape)
    print("Shape of X_validation:", X_val.shape)
    print("Shape of Y_validation:", Y_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of Y_test:", Y_test.shape)

    return X_train, Y_train, X_val, Y_val, seq_dirs_train, seq_dirs_validation, seq_dirs_test

# ==========================
# MODELS
# ==========================
def rnn_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(inputs)
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(x)
    x = tf.keras.layers.SimpleRNN(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(datasize)(x)
    # outputs = tf.keras.layers.Dense(datasize, activation="relu")(x)
    outputs = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)
    return tf.keras.Model(inputs, outputs)

def lstm_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(datasize)(x)
    outputs = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)
    # outputs = tf.keras.layers.Dense(datasize, activation="softplus")(x)
    # outputs = tf.keras.layers.Dense(datasize, activation="relu")(x)
    return tf.keras.Model(inputs, outputs)

def tcn_model(window_size, datasize, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(window_size, datasize))

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(datasize)(x)
    outputs = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return tf.keras.Model(inputs, outputs)

def transformer_model(window_size, datasize, num_heads=4, ff_dim=64, num_layers=3, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(window_size, datasize))

    def positional_encoding(length, depth):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos, dtype=tf.float32)

    pos = positional_encoding(window_size, datasize)
    x = inputs + pos[tf.newaxis, :, :]

    def create_causal_mask(seq_len):
        return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    causal_mask = create_causal_mask(window_size)

    for _ in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=datasize // num_heads
        )(x, x, attention_mask=causal_mask)
        attn = tf.keras.layers.Dropout(dropout_rate)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(datasize),
            tf.keras.layers.Dropout(dropout_rate)
        ])(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    out = tf.keras.layers.Dense(datasize)(x[:, -1, :])
    out = tf.keras.layers.LeakyReLU(alpha=0.01)(out)
    out = tf.keras.layers.ReLU()(out)
    return tf.keras.Model(inputs, out)

def build_model(model_type, window_size, datasize):
    if model_type == "rnn":
        model = rnn_model(window_size, datasize)
        batch_size = 16
    elif model_type == "lstm":
        model = lstm_model(window_size, datasize)
        batch_size = 16
    elif model_type == "tcn":
        model = tcn_model(window_size, datasize)
        batch_size = 16
    elif model_type == "transformer":
        model = transformer_model(window_size, datasize)
        batch_size = 16
    else:
        raise ValueError("model_type must be in: rnn, lstm, tcn, transformer")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mae")
    model.summary()
    return model, batch_size

# ==========================
# TRAIN (NO LOSS FIGURE OUTPUT)
# ==========================
def train_model_no_files(model, X_train, Y_train, X_val, Y_val, batch_size, model_type):
    print(f"[TRAIN] {model_type.upper()} ...")
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(X_val, Y_val),
        verbose=1
    )
    return history

# ==========================
# PREDICTION (AUTO-RÉGRESSIVE SUR DONNÉES NORMALISÉES)
# ==========================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    seq_norm: (T, datasize) normalisé
    Retourne preds_norm: (output_size, datasize) (les sorties brutes du modèle)
    Mais on renormalise la fenêtre avec pred/sum(pred) pour l'entrée suivante.
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

def predict_testset_normalized(model, sequences_array, sequences_sums_array,
                               seq_dirs_test, sequence_dirs,
                               window_size, output_size):
    """
    Pour chaque séquence test :
      - calcule les prédictions normalisées brutes du modèle ;
      - compare directement ces prédictions aux fréquences normalisées réelles ;
      - n'applique ni dénormalisation ni écrêtage des valeurs négatives.

    Remarque : sequences_sums_array est conservé dans la signature afin de ne pas
    modifier la structure générale du pipeline, mais il n'est pas utilisé ici.
    """
    preds_norm_all = []
    y_norm_all = []
    valid_seq_names = []

    for seq_name in seq_dirs_test:
        idx = sequence_dirs.index(seq_name)
        seq_norm = sequences_array[idx]

        if len(seq_norm) < window_size + output_size:
            print(f"[TESTSET] {seq_name} trop courte -> ignorée.")
            continue

        preds_norm = autoregressive_predict_norm(
            model, seq_norm, window_size, output_size
        )

        y_norm = seq_norm[window_size:window_size + output_size]

        preds_norm_all.append(preds_norm)
        y_norm_all.append(y_norm)
        valid_seq_names.append(seq_name)

    return (
        np.array(preds_norm_all),
        np.array(y_norm_all),
        valid_seq_names,
    )

def predict_new_sequences_normalized(model, new_sequences_root,
                                     window_size, output_size):
    """
    Parcourt new_sequences_root et retourne directement :
      preds_norm : (N, output_size, datasize)
      y_norm     : (N, output_size, datasize)
      seq_names  : liste
      file_names : dict {seq_name: [files...]}

    Aucune dénormalisation et aucun np.maximum ne sont appliqués.
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
            print(f"[NEW] {seq_name} trop courte -> ignorée.")
            continue

        preds_norm = autoregressive_predict_norm(
            model, seq_norm, window_size, output_size
        )

        y_norm = seq_norm[window_size:window_size + output_size]

        preds_list.append(preds_norm)
        y_list.append(y_norm)
        seq_names.append(seq_name)

    if len(seq_names) == 0:
        return None, None, [], file_names_map

    return np.array(preds_list), np.array(y_list), seq_names, file_names_map

# ==========================
# ERRORS (PRINT)
# ==========================
def print_errors_global_and_per_sequence(preds, y_true, seq_names, label, model_type):
    """
    Affiche:
      - global MAE/MSE/RMSE/MRE/R2
      - par séquence
      - moyenne par séquence
    """
    print("\n" + "="*80)
    print(f"[ERREURS] {label} — {model_type.upper()}")
    print("="*80)

    if preds is None or len(preds) == 0:
        print("Aucune prédiction disponible.")
        return

    # Global
    diff = preds - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    mask = y_true != 0
    if np.any(mask):
        mre = float(np.mean(np.abs((y_true[mask] - preds[mask]) / y_true[mask])) * 100)
    else:
        mre = 0.0

    r2 = float(r2_score(y_true.flatten(), preds.flatten()))

    print(f"\nGlobal:")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MRE : {mre:.2f}%")
    print(f"  R²  : {r2:.4f}")

    # Per sequence
    mae_list, mse_list, rmse_list, mre_list, r2_list = [], [], [], [], []

    print(f"\nPar séquence:")
    for i, name in enumerate(seq_names):
        yt = y_true[i]
        yp = preds[i]
        d = yp - yt

        mae_i = float(np.mean(np.abs(d)))
        mse_i = float(np.mean(d**2))
        rmse_i = float(np.sqrt(mse_i))

        mask_i = yt != 0
        if np.any(mask_i):
            rel = np.abs((yt[mask_i] - yp[mask_i]) / yt[mask_i])
            mre_i = float(np.sum(rel) / yt.size * 100)   # même formule agrégée
        else:
            mre_i = 0.0

        r2_i = float(r2_score(yt.flatten(), yp.flatten()))

        mae_list.append(mae_i)
        mse_list.append(mse_i)
        rmse_list.append(rmse_i)
        mre_list.append(mre_i)
        r2_list.append(r2_i)

        print(f"  - {name}: MAE={mae_i:.4f} | MSE={mse_i:.4f} | RMSE={rmse_i:.4f} | MRE={mre_i:.2f}% | R²={r2_i:.4f}")

    print(f"\nMoyenne sur séquences:")
    print(f"  mean MAE : {float(np.mean(mae_list)):.4f}")
    print(f"  mean MSE : {float(np.mean(mse_list)):.4f}")
    print(f"  mean RMSE: {float(np.mean(rmse_list)):.4f}")
    print(f"  mean MRE : {float(np.mean(mre_list)):.2f}%")
    print(f"  mean R²  : {float(np.mean(r2_list)):.4f}")

# ==========================
# COMBINED FIGURES (BARS)
# ==========================
def generate_combined_figures_generic(all_predictions,
                                      all_y_test,
                                      seq_names,
                                      output_size,
                                      datasize,
                                      out_root,
                                      model_types,
                                      window_size,
                                      xmax,
                                      yscale_mode="per_timestep",
                                      fixed_ylim=None):
    """
    all_predictions[model_type] = (N, output_size, datasize)
    all_y_test[model_type]      = (N, output_size, datasize)   (identique pour tous les modèles)
    seq_names = liste de N noms (ordre)
    out_root = dossier où écrire (ex: base_output_dir/figures_all)
    """

    safe_makedirs(out_root)

    # Axe X "physique"
    x_min, x_max = 0.0, xmax
    bin_edges = np.linspace(x_min, x_max, datasize + 1)
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = (x_max - x_min) / datasize

    x_groups = x_centers
    total_bars = len(model_types) + 1
    bar_width = bin_width * 0.8 / total_bars
    edge_pad = bin_width * 0.1

    colors = {
        "y_test": "#444444",
        "rnn": "green",
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

        # y_limit per sequence if needed
        if yscale_mode == "per_sequence":
            seq_max = float(np.max(all_y_test[first_model][seq_idx]))
            for mt in model_types:
                seq_max = max(seq_max, float(np.max(all_predictions[mt][seq_idx])))
            seq_y_limit = seq_max * 1.1 if seq_max > 0 else 1.0

        for t in range(output_size):

            if yscale_mode == "per_timestep":
                ts_max = float(np.max(all_y_test[first_model][seq_idx, t, :]))
                for mt in model_types:
                    ts_max = max(ts_max, float(np.max(all_predictions[mt][seq_idx, t, :])))
                y_limit = ts_max * 1.1 if ts_max > 0 else 1.0
            elif yscale_mode == "per_sequence":
                y_limit = seq_y_limit
            elif yscale_mode == "fixed":
                if fixed_ylim is None or fixed_ylim <= 0:
                    raise ValueError("fixed_ylim doit être > 0 si yscale_mode='fixed'")
                y_limit = float(fixed_ylim)
            else:
                raise ValueError("yscale_mode must be per_timestep / per_sequence / fixed")

            plt.figure(figsize=(16, 10))

            for bin_idx, x_val in enumerate(x_groups):
                # y_test bar
                for model_offset, key in enumerate(["y_test"] + model_types):
                    if key == "y_test":
                        y = all_y_test[first_model][seq_idx, t, bin_idx]
                    else:
                        y = all_predictions[key][seq_idx, t, bin_idx]

                    offset = (model_offset - (total_bars - 1) / 2.0) * bar_width
                    x_pos = x_val + offset

                    plt.bar(
                        x_pos,
                        y,
                        width=bar_width,
                        color=colors.get(key, "gray"),
                        label=custom_labels.get(key, key.upper()) if bin_idx == 0 else ""
                    )

            plt.xlim(x_min - edge_pad, x_max + edge_pad)
            plt.margins(x=0)
            y_min = min(
                0,
                np.min([
                    np.min(all_predictions[mt][seq_idx, t, :])
                    for mt in model_types
                ])
            )

            plt.ylim(y_min * 1.1, y_limit)
            # plt.ylim(0, y_limit)
            plt.xlabel("Grain Size (mm)", fontsize=19)
            plt.ylabel("Frequency", fontsize=19)
            plt.title(f"{seq_name} — Time Step {t + window_size + 1}")

            plt.xticks(x_groups[::2], [f"{x:.3f}" for x in x_groups[::2]], rotation=45)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()

            fig_path = os.path.join(seq_dir, f"{t + window_size}.png")
            plt.savefig(fig_path)
            plt.close()

# ==========================
# MAIN
# ==========================
def main():
    print("="*80)
    print("PIPELINE MINIMAL (MODEL + FIGURES COMBINÉES) + PRINT ERREURS")
    print("="*80)

    # Validate models
    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix: {AVAILABLE_MODELS}")

    # 1) Load data (memory only)
    print("\n[1] Chargement + normalisation (en mémoire)...")
    sequences_array, sequences_sums_array, file_names, sequence_dirs = load_and_normalize_data_in_memory(data_dir)

    # 2) Datasets split
    print("\n[2] Split train/val/test ...")
    X_train, Y_train, X_val, Y_val, seq_dirs_train, seq_dirs_val, seq_dirs_test = prepare_datasets(
        sequences_array, sequences_sums_array, sequence_dirs, window_size
    )
    print(f"  Train seqs: {len(seq_dirs_train)}")
    print(f"  Val   seqs: {len(seq_dirs_val)}")
    print(f"  Test  seqs: {len(seq_dirs_test)}")

    # Output dirs we keep
    figures_test_root = os.path.join(base_output_dir, "figures_all")
    figures_new_root  = os.path.join(base_output_dir, "figures_all_new_sequences2")
    reset_directory(figures_test_root)
    reset_directory(figures_new_root)

    # containers for combined figures
    all_predictions_test = {}
    all_y_test = {}

    all_predictions_new = {}
    all_y_new = {}
    new_seq_names_ref = None  # liste commune (ordre)

    # 3) Loop models
    for model_type in model_types:
        print("\n" + "="*80)
        print(f"[MODEL] {model_type.upper()}")
        print("="*80)

        # Build + train
        model, batch_size = build_model(model_type, window_size, datasize)
        _ = train_model_no_files(model, X_train, Y_train, X_val, Y_val, batch_size, model_type)

        # --- TESTSET predictions normalisées ---
        print(f"\n[TESTSET] Prédictions normalisées ({model_type.upper()}) ...")
        preds_test, y_test, valid_test_names = predict_testset_normalized(
            model, sequences_array, sequences_sums_array,
            seq_dirs_test, sequence_dirs,
            window_size, output_size
        )

        # Store for combined figures
        all_predictions_test[model_type] = preds_test
        all_y_test[model_type] = y_test

        # Print errors testset
        print_errors_global_and_per_sequence(
            preds_test, y_test, valid_test_names,
            label="TEST SET (split)", model_type=model_type
        )

        # --- NEW SEQUENCES predictions normalisées ---
        if os.path.exists(new_sequences_root):
            print(f"\n[NEW] Prédictions normalisées ({model_type.upper()}) ...")
            preds_new, y_new, new_names, _ = predict_new_sequences_normalized(
                model, new_sequences_root, window_size, output_size
            )

            if preds_new is not None and len(new_names) > 0:
                # store for combined figures
                all_predictions_new[model_type] = preds_new
                all_y_new[model_type] = y_new

                if new_seq_names_ref is None:
                    new_seq_names_ref = new_names
                else:
                    # sécurité: on force le même ordre
                    if new_names != new_seq_names_ref:
                        print("[WARN] ordre des nouvelles séquences différent selon modèle. Vérifie les dossiers.")
                        new_seq_names_ref = new_names

                # Print errors new sequences
                print_errors_global_and_per_sequence(
                    preds_new, y_new, new_names,
                    label="NEW SEQUENCES", model_type=model_type
                )
            else:
                print("[NEW] Aucune nouvelle séquence valide (vide ou trop courte).")
        else:
            print(f"[NEW] Dossier introuvable: {new_sequences_root}")

        # Save model (ONLY MODEL FILE)
        print(f"\n[SAVE] Sauvegarde du modèle {model_type.upper()} ...")
        model_path = os.path.join(base_output_dir, f"model_{model_type}.keras")
        model.save(model_path)
        print(f"  ✅ Modèle sauvegardé: {model_path}")

        tf.keras.backend.clear_session()

    # 4) Combined figures TESTSET
    print("\n" + "="*80)
    print("[FIGURES] Génération des figures combinées — TEST SET")
    print("="*80)

    # si un seul modèle, ça marche aussi (ground truth + 1 prédiction)
    # generate_combined_figures_generic(
    #     all_predictions=all_predictions_test,
    #     all_y_test=all_y_test,
    #     seq_names=list(all_y_test[model_types[0]].shape[0] and valid_test_names),
    #     output_size=output_size,
    #     datasize=datasize,
    #     out_root=figures_test_root,
    #     model_types=model_types,
    #     window_size=window_size,
    #     xmax=xmax,
    #     yscale_mode="per_timestep"  # lisible (change en "per_sequence" si tu veux)
    # )
    # print(f"  ✅ Figures combinées test-set dans: {figures_test_root}")

    # 5) Combined figures NEW SEQUENCES
    if new_seq_names_ref is not None and len(new_seq_names_ref) > 0:
        print("\n" + "="*80)
        print("[FIGURES] Génération des figures combinées — NEW SEQUENCES")
        print("="*80)

        # generate_combined_figures_generic(
        #     all_predictions=all_predictions_new,
        #     all_y_test=all_y_new,
        #     seq_names=new_seq_names_ref,
        #     output_size=output_size,
        #     datasize=datasize,
        #     out_root=figures_new_root,
        #     model_types=model_types,
        #     window_size=window_size,
        #     xmax=xmax,
        #     yscale_mode="per_timestep"
        # )
        # print(f"  ✅ Figures combinées new sequences dans: {figures_new_root}")
    else:
        print("\n[FIGURES] Aucune nouvelle séquence valide -> pas de figures new_sequences.")

    print("\n" + "="*80)
    print("FIN. Sorties générées:")
    print("  - model_<model>.keras")
    print("  - figures_all/ (combined test-set)")
    print("  - figures_all_new_sequences/ (combined new sequences)")
    print("="*80)

if __name__ == "__main__":
    main()
