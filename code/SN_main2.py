# This pipeline performs 10-fold cross-validation at the sequence level.
# It trains and evaluates the selected model and saves each fold model and the CSV summaries.

# Suppress TensorFlow warnings before importing TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import csv
import shutil
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

# ============================================================
# CONFIGURATION
# ============================================================
seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

data_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/trm"
base_output_dir = "/home/admin-eyounes/Desktop/29_janv/1/données+code2/article/output_kfold"

datasize = 30
window_size = 5
output_size = 55

N_SPLITS = 10
INNER_VAL_SIZE = 0.10  # 10% of the remaining nine folds is used for validation

AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]
model_types = ["transformer"]  # Change here if needed: ["rnn", "lstm", "tcn", "transformer"]

os.makedirs(base_output_dir, exist_ok=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def numerical_sort(value):
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p for p in parts]

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed_value: int):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# ============================================================
# DATA LOADING
# ============================================================
def load_and_normalize_data_in_memory(data_dir: str):
    """
    Reads each subdirectory of data_dir as one sequence.
    For each .txt file:
      - vals contains the raw distribution
      - sums contains the sum of vals
      - normalized_values is computed as vals / sum(vals)

    Returns:
      sequences_array: shape (N_sequences, T, datasize)
      sequences_sums_array: shape (N_sequences, T)
      file_names
      sequence_dirs
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
                next(f)  # Skip the header
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
    print("Nombre de séquences:", len(sequence_dirs))

    return sequences_array, sequences_sums_array, file_names, sequence_dirs

# ============================================================
# SLIDING WINDOWS
# ============================================================
def create_sliding_windows(sequence, sums, window_size):
    X, Y, Y_sums = [], [], []

    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        Y.append(sequence[i + window_size])
        Y_sums.append(sums[i + window_size])

    return np.array(X), np.array(Y), np.array(Y_sums)

def build_dataset_from_indices(sequences_array, sequences_sums_array, indices, window_size):
    """
    Builds X and Y from a list of sequence indices.
    Important: the split is performed by sequence, not by window.
    """
    X_all, Y_all = [], []

    for seq_idx in indices:
        X_seq, Y_seq, _ = create_sliding_windows(
            sequences_array[seq_idx],
            sequences_sums_array[seq_idx],
            window_size
        )

        if len(X_seq) > 0:
            X_all.append(X_seq)
            Y_all.append(Y_seq)

    if len(X_all) == 0:
        raise ValueError("Aucune fenêtre créée. Vérifie window_size et longueur des séquences.")

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all

# ============================================================
# MODELS
# ============================================================
def rnn_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))

    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(inputs)
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(x)
    x = tf.keras.layers.SimpleRNN(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(datasize)(x)

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

    return tf.keras.Model(inputs, outputs)

def tcn_model(window_size, datasize, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(window_size, datasize))

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="causal", dilation_rate=4)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(datasize)(x)

    outputs = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)

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
            tf.keras.layers.LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dense(datasize),
            tf.keras.layers.Dropout(dropout_rate)
        ])(x)

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    out = tf.keras.layers.Dense(datasize)(x[:, -1, :])
    out = tf.keras.layers.LeakyReLU(negative_slope=0.01)(out)

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

    return model, batch_size

# ============================================================
# TRAINING
# ============================================================
def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, model_type, fold_id):
    print(f"\n[TRAIN] {model_type.upper()} — Fold {fold_id:02d}")

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(X_val, Y_val),
        verbose=1
    )

    return history

# ============================================================
# AUTOREGRESSIVE PREDICTION
# ============================================================
def autoregressive_predict_norm(model, seq_norm, window_size, output_size):
    """
    seq_norm: normalized array with shape (T, datasize)
    Returns preds_norm with shape (output_size, datasize).

    The next window is updated using the renormalized prediction.
    """
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

def predict_and_denormalize_test_indices(
    model,
    sequences_array,
    sequences_sums_array,
    test_indices,
    sequence_dirs,
    window_size,
    output_size
):
    """
    For each test sequence:
      - performs normalized autoregressive prediction
      - denormalizes the prediction using actual_sums
      - compares it with the denormalized ground truth
    """
    denorm_preds_all = []
    denorm_y_all = []
    valid_seq_names = []

    for seq_idx in test_indices:
        seq_name = sequence_dirs[seq_idx]
        seq_norm = sequences_array[seq_idx]
        sums = sequences_sums_array[seq_idx]

        if len(seq_norm) < window_size + output_size:
            print(f"[TEST] {seq_name} trop courte -> ignorée.")
            continue

        preds_norm = autoregressive_predict_norm(
            model,
            seq_norm,
            window_size,
            output_size
        )

        actual_sums = sums[window_size:window_size + output_size]

        denorm_preds = preds_norm * actual_sums[:, np.newaxis]
        denorm_y = seq_norm[window_size:window_size + output_size] * actual_sums[:, np.newaxis]

        denorm_preds = np.maximum(denorm_preds, 0)

        denorm_preds_all.append(denorm_preds)
        denorm_y_all.append(denorm_y)
        valid_seq_names.append(seq_name)

    if len(valid_seq_names) == 0:
        return None, None, []

    return np.array(denorm_preds_all), np.array(denorm_y_all), valid_seq_names

# ============================================================
# METRICS
# ============================================================
def compute_global_metrics(denorm_preds, denorm_y):
    diff = denorm_preds - denorm_y

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    mask = denorm_y != 0

    if np.any(mask):
        mre = float(np.mean(np.abs((denorm_y[mask] - denorm_preds[mask]) / denorm_y[mask])) * 100)
    else:
        mre = 0.0

    r2 = float(r2_score(denorm_y.flatten(), denorm_preds.flatten()))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MRE": mre,
        "R2": r2
    }

def compute_per_sequence_metrics(denorm_preds, denorm_y, seq_names):
    rows = []

    for i, name in enumerate(seq_names):
        yt = denorm_y[i]
        yp = denorm_preds[i]

        diff = yp - yt

        mae_i = float(np.mean(np.abs(diff)))
        mse_i = float(np.mean(diff ** 2))
        rmse_i = float(np.sqrt(mse_i))

        # Same approach as in the previous code:
        # rel is computed only where yt > 0 and yp >= 0,
        # and is then divided by yt.size.
        mask_i = (yt > 0) & (yp >= 0)

        if np.any(mask_i):
            rel = np.abs((yt[mask_i] - yp[mask_i]) / yt[mask_i])
            mre_i = float(np.sum(rel) / yt.size * 100)
        else:
            mre_i = 0.0

        r2_i = float(r2_score(yt.flatten(), yp.flatten()))

        rows.append({
            "sequence": name,
            "MAE": mae_i,
            "MSE": mse_i,
            "RMSE": rmse_i,
            "MRE": mre_i,
            "R2": r2_i
        })

    return rows

def print_fold_errors(denorm_preds, denorm_y, seq_names, model_type, fold_id):
    print("\n" + "=" * 80)
    print(f"[ERREURS] TEST SET — {model_type.upper()} — FOLD {fold_id:02d}")
    print("=" * 80)

    if denorm_preds is None or len(denorm_preds) == 0:
        print("Aucune prédiction disponible.")
        return None, []

    global_metrics = compute_global_metrics(denorm_preds, denorm_y)
    per_seq_rows = compute_per_sequence_metrics(denorm_preds, denorm_y, seq_names)

    print("\nGlobal:")
    print(f"  MAE : {global_metrics['MAE']:.4f}")
    print(f"  MSE : {global_metrics['MSE']:.4f}")
    print(f"  RMSE: {global_metrics['RMSE']:.4f}")
    print(f"  MRE : {global_metrics['MRE']:.2f}%")
    print(f"  R²  : {global_metrics['R2']:.4f}")

    print("\nPar séquence:")
    for row in per_seq_rows:
        print(
            f"  - {row['sequence']}: "
            f"MAE={row['MAE']:.4f} | "
            f"MSE={row['MSE']:.4f} | "
            f"RMSE={row['RMSE']:.4f} | "
            f"MRE={row['MRE']:.2f}% | "
            f"R²={row['R2']:.4f}"
        )

    mean_mre_seq = float(np.mean([r["MRE"] for r in per_seq_rows]))

    print("\nMoyenne sur séquences:")
    print(f"  mean MRE : {mean_mre_seq:.2f}%")

    return global_metrics, per_seq_rows

# ============================================================
# CSV OUTPUT
# ============================================================
def save_kfold_summary_csv(summary_rows, output_csv):
    fieldnames = [
        "model_type",
        "fold",
        "n_train_sequences",
        "n_val_sequences",
        "n_test_sequences",
        "global_MAE",
        "global_MSE",
        "global_RMSE",
        "global_MRE",
        "global_R2",
        "mean_sequence_MRE"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print(f"\n✅ Résumé CSV sauvegardé: {output_csv}")

def save_per_sequence_csv(per_sequence_all_rows, output_csv):
    fieldnames = [
        "model_type",
        "fold",
        "sequence",
        "MAE",
        "MSE",
        "RMSE",
        "MRE",
        "R2"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in per_sequence_all_rows:
            writer.writerow(row)

    print(f"✅ Résumé par séquence sauvegardé: {output_csv}")

# ============================================================
# MAIN K-FOLD PIPELINE
# ============================================================
def main():
    print("=" * 80)
    print("PIPELINE 10-FOLD CROSS-VALIDATION PAR SÉQUENCE")
    print("=" * 80)

    for mt in model_types:
        if mt not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {mt} non valide. Choix possibles: {AVAILABLE_MODELS}")

    # 1) Load the data
    print("\n[1] Chargement + normalisation des données...")
    sequences_array, sequences_sums_array, file_names, sequence_dirs = load_and_normalize_data_in_memory(data_dir)

    n_sequences = len(sequence_dirs)

    if n_sequences < N_SPLITS:
        raise ValueError(f"Nombre de séquences ({n_sequences}) inférieur au nombre de folds ({N_SPLITS}).")

    all_indices = np.arange(n_sequences)

    # 2) Perform K-fold splitting by sequence
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    # 3) Loop over the models
    for model_type in model_types:
        print("\n" + "#" * 80)
        print(f"# MODEL: {model_type.upper()}")
        print("#" * 80)

        model_output_dir = os.path.join(base_output_dir, f"kfold_{model_type}")
        reset_directory(model_output_dir)

        summary_rows = []
        per_sequence_all_rows = []

        fold_global_mre_values = []
        fold_mean_seq_mre_values = []

        # 4) Loop over the folds
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(all_indices), start=1):

            print("\n" + "=" * 80)
            print(f"[FOLD {fold_id:02d}/{N_SPLITS}]")
            print("=" * 80)

            # Use a different but reproducible seed for each fold
            fold_seed = seed + fold_id
            set_all_seeds(fold_seed)

            # Internal TRAIN/VALIDATION split from the remaining nine folds
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=INNER_VAL_SIZE,
                random_state=fold_seed,
                shuffle=True
            )

            seq_dirs_train = [sequence_dirs[i] for i in train_idx]
            seq_dirs_val = [sequence_dirs[i] for i in val_idx]
            seq_dirs_test = [sequence_dirs[i] for i in test_idx]

            print(f"Train sequences: {len(seq_dirs_train)}")
            print(f"Val sequences  : {len(seq_dirs_val)}")
            print(f"Test sequences : {len(seq_dirs_test)}")

            print("\nTest sequences:")
            for s in seq_dirs_test:
                print(f"  - {s}")

            # 5) Build the sliding-window datasets
            print("\n[DATASET] Création des fenêtres...")
            X_train, Y_train = build_dataset_from_indices(
                sequences_array,
                sequences_sums_array,
                train_idx,
                window_size
            )

            X_val, Y_val = build_dataset_from_indices(
                sequences_array,
                sequences_sums_array,
                val_idx,
                window_size
            )

            print("Shape of X_train:", X_train.shape)
            print("Shape of Y_train:", Y_train.shape)
            print("Shape of X_val  :", X_val.shape)
            print("Shape of Y_val  :", Y_val.shape)

            # 6) Build and train the model
            print("\n[MODEL] Build...")
            model, batch_size = build_model(model_type, window_size, datasize)

            _ = train_model(
                model,
                X_train,
                Y_train,
                X_val,
                Y_val,
                batch_size,
                model_type,
                fold_id
            )

            # 7) Predict the test fold
            print(f"\n[TEST] Prédictions fold {fold_id:02d}...")
            denorm_preds_test, denorm_y_test, valid_test_names = predict_and_denormalize_test_indices(
                model,
                sequences_array,
                sequences_sums_array,
                test_idx,
                sequence_dirs,
                window_size,
                output_size
            )

            # 8) Compute errors
            global_metrics, per_seq_rows = print_fold_errors(
                denorm_preds_test,
                denorm_y_test,
                valid_test_names,
                model_type,
                fold_id
            )

            if global_metrics is not None:
                mean_sequence_mre = float(np.mean([r["MRE"] for r in per_seq_rows]))

                fold_global_mre_values.append(global_metrics["MRE"])
                fold_mean_seq_mre_values.append(mean_sequence_mre)

                summary_rows.append({
                    "model_type": model_type,
                    "fold": fold_id,
                    "n_train_sequences": len(seq_dirs_train),
                    "n_val_sequences": len(seq_dirs_val),
                    "n_test_sequences": len(seq_dirs_test),
                    "global_MAE": global_metrics["MAE"],
                    "global_MSE": global_metrics["MSE"],
                    "global_RMSE": global_metrics["RMSE"],
                    "global_MRE": global_metrics["MRE"],
                    "global_R2": global_metrics["R2"],
                    "mean_sequence_MRE": mean_sequence_mre
                })

                for row in per_seq_rows:
                    row_copy = {
                        "model_type": model_type,
                        "fold": fold_id,
                        "sequence": row["sequence"],
                        "MAE": row["MAE"],
                        "MSE": row["MSE"],
                        "RMSE": row["RMSE"],
                        "MRE": row["MRE"],
                        "R2": row["R2"]
                    }
                    per_sequence_all_rows.append(row_copy)

            # 9) Save the model for this fold
            model_path = os.path.join(
                model_output_dir,
                f"model_{model_type}_fold{fold_id:02d}.keras"
            )

            model.save(model_path)
            print(f"\n✅ Modèle sauvegardé: {model_path}")

            # Clear memory
            tf.keras.backend.clear_session()
            del model

        # 10) Display the final summary for this model
        print("\n" + "#" * 80)
        print(f"# FINAL 10-FOLD SUMMARY — {model_type.upper()}")
        print("#" * 80)

        fold_global_mre_values = np.array(fold_global_mre_values, dtype=float)
        fold_mean_seq_mre_values = np.array(fold_mean_seq_mre_values, dtype=float)

        print("\nMRE global par fold:")
        for i, val in enumerate(fold_global_mre_values, start=1):
            print(f"  Fold {i:02d}: {val:.2f}%")

        print("\nMRE moyen par séquence par fold:")
        for i, val in enumerate(fold_mean_seq_mre_values, start=1):
            print(f"  Fold {i:02d}: {val:.2f}%")

        print("\nRésultat à rapporter dans l'article:")
        print(
            f"  Global MRE = "
            f"{np.mean(fold_global_mre_values):.2f} ± {np.std(fold_global_mre_values, ddof=1):.2f}%"
        )

        print(
            f"  Mean sequence MRE = "
            f"{np.mean(fold_mean_seq_mre_values):.2f} ± {np.std(fold_mean_seq_mre_values, ddof=1):.2f}%"
        )

        # 11) Save the CSV summaries
        summary_csv = os.path.join(model_output_dir, f"kfold_summary_{model_type}.csv")
        per_seq_csv = os.path.join(model_output_dir, f"kfold_per_sequence_{model_type}.csv")

        save_kfold_summary_csv(summary_rows, summary_csv)
        save_per_sequence_csv(per_sequence_all_rows, per_seq_csv)

    print("\n" + "=" * 80)
    print("FIN DU 10-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"Sorties sauvegardées dans: {base_output_dir}")

if __name__ == "__main__":
    main()