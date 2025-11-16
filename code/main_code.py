import os
import numpy as np
import re
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from uuid import uuid4

import matplotlib as mpl
mpl.rcParams.update({
    # Résolution
    "figure.dpi": 220,      # rendu à l'écran
    "savefig.dpi": 300,     # rendu dans les PNG sauvegardés
    # Tailles de police globales
    "font.size": 18,        # taille de base
    "axes.titlesize": 28,   # titre de la figure
    "axes.labelsize": 26,   # labels X/Y
    "xtick.labelsize": 22,  # ticks X
    "ytick.labelsize": 22,  # ticks Y
    "legend.fontsize": 22,  # légende
    # Marges
    "figure.autolayout": True,  # équivaut souvent à tight_layout()
    # Épaisseurs par défaut
    "lines.linewidth": 2.5,
})

xmax=0.12
# Pour assurer la reproductibilité
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# A mettre à jour avec les bonnes adresses
data_dir = "towards trm/ directory"
base_output_dir = "towards output/ directory "
new_sequences_root = "towards trm/test_sequences/ directory"


datasize = 30  # nombre de fréquences par fichier
window_size =5   # Taille de la fenêtre d'entrée
output_size =55  # Nombre de pas de temps à prédire

# Liste des modèles possibles (configurable)
AVAILABLE_MODELS = ["transformer", "rnn", "lstm", "tcn"]

# Choisir les modèles à exécuter (modifie cette liste selon ton besoin)
model_types =["transformer", "lstm","rnn", "tcn"]
# Fonction pour supprimer et recréer un répertoire
def reset_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Fonction pour trier numériquement les noms de fichiers
def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

# Lecture et normalisation des fichiers
def load_and_normalize_data(normalized_output_dir):
    reset_directory(normalized_output_dir)

    sequences = []
    file_names = []
    sequence_dirs = []
    sequences_sums = []

    for sequence_dir in os.listdir(data_dir):
        sequence_path = os.path.join(data_dir, sequence_dir)
        if os.path.isdir(sequence_path):
            normalized_sequence_dir = os.path.join(normalized_output_dir, sequence_dir)
            os.makedirs(normalized_sequence_dir, exist_ok=True)
            
            sequence = []
            sequence_sums = []
            sequence_files = []
            
            for file_name in sorted(os.listdir(sequence_path), key=numerical_sort):
                file_path = os.path.join(sequence_path, file_name)
                if file_name.endswith('.txt'):
                    with open(file_path, 'r') as file:
                        next(file)
                        file_content = file.read().strip().split()
                        file_content_float = [float(val) for val in file_content]
                        
                        sum_value = sum(file_content_float)
                        sequence_sums.append(sum_value)
                        
                        if sum_value != 0:
                            normalized_values = [round(val / sum_value, 16) for val in file_content_float]
                        else:
                            normalized_values = [0 for _ in file_content_float]

                        # if sum_value != 0:
                        #     normalized_values = [val / sum_value for val in file_content_float]
                        # else:
                        #     normalized_values = [0 for _ in file_content_float]
                        
                        sequence.append(normalized_values)
                        sequence_files.append(file_name)
                        
                        normalized_file_path = os.path.join(normalized_sequence_dir, file_name)
                        # with open(normalized_file_path, 'w') as norm_file:
                        #     norm_file.write('\n'.join(map(str, normalized_values)))
                        with open(normalized_file_path, 'w') as norm_file:
                            norm_file.write('\n'.join(f"{val:.8f}" for val in normalized_values))

            
            sequences.append(sequence)
            file_names.append(sequence_files)
            sequences_sums.append(sequence_sums)
            sequence_dirs.append(sequence_dir)

    sequences_array = np.array([np.array(seq) for seq in sequences])
    sequences_sums_array = np.array(sequences_sums)
    print("Shape of sequences_array:", sequences_array.shape)
    print("Shape of sequences_sums_array:", sequences_sums_array.shape)
    
    return sequences_array, sequences_sums_array, file_names, sequence_dirs

# Fonction pour créer des paires de fenêtres glissantes
def create_sliding_windows(sequence, sums, window_size):
    X, Y, Y_sums = [], [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        Y.append(sequence[i + window_size])
        Y_sums.append(sums[i + window_size])
    return np.array(X), np.array(Y), np.array(Y_sums)

# Création des datasets
def prepare_datasets(sequences_array, sequences_sums_array, sequence_dirs):
    X_all, Y_all, Y_sums_all = [], [], []
    for seq_idx in range(sequences_array.shape[0]):
        X_seq, Y_seq, Y_sums_seq = create_sliding_windows(
            sequences_array[seq_idx], sequences_sums_array[seq_idx], window_size
        )
        X_all.append(X_seq)
        Y_all.append(Y_seq)
        Y_sums_all.append(Y_sums_seq)

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    Y_sums_all = np.concatenate(Y_sums_all, axis=0)

    train_idx, val_test_idx = train_test_split(
        range(len(sequence_dirs)), test_size=0.2, random_state=43
    )
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=0.25, random_state=42
    )

    seq_dirs_train = [sequence_dirs[i] for i in train_idx]
    seq_dirs_validation = [sequence_dirs[i] for i in val_idx]
    seq_dirs_test = [sequence_dirs[i] for i in test_idx]

    X_train, Y_train, Y_sums_train = [], [], []
    X_validation, Y_validation, Y_sums_validation = [], [], []
    X_test, Y_test, Y_sums_test = [], [], []

    for seq_idx in range(sequences_array.shape[0]):
        X_seq, Y_seq, Y_sums_seq = create_sliding_windows(
            sequences_array[seq_idx], sequences_sums_array[seq_idx], window_size
        )
        if sequence_dirs[seq_idx] in seq_dirs_train:
            X_train.append(X_seq)
            Y_train.append(Y_seq)
            Y_sums_train.append(Y_sums_seq)
        elif sequence_dirs[seq_idx] in seq_dirs_validation:
            X_validation.append(X_seq)
            Y_validation.append(Y_seq)
            Y_sums_validation.append(Y_sums_seq)
        elif sequence_dirs[seq_idx] in seq_dirs_test:
            X_test.append(X_seq)
            Y_test.append(Y_seq)
            Y_sums_test.append(Y_sums_seq)

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    Y_sums_train = np.concatenate(Y_sums_train, axis=0)
    X_validation = np.concatenate(X_validation, axis=0)
    Y_validation = np.concatenate(Y_validation, axis=0)
    Y_sums_validation = np.concatenate(Y_sums_validation, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    Y_sums_test = np.concatenate(Y_sums_test, axis=0)

    print("Shape of X_train:", X_train.shape)
    print("Shape of Y_train:", Y_train.shape)
    print("Shape of X_validation:", X_validation.shape)
    print("Shape of Y_validation:", Y_validation.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of Y_test:", Y_test.shape)

    return (X_train, Y_train, Y_sums_train, X_validation, Y_validation,
            Y_sums_validation, X_test, Y_test, Y_sums_test, seq_dirs_train,
            seq_dirs_validation, seq_dirs_test)

# Définition des modèles
def rnn_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))
    rnn1 = tf.keras.layers.SimpleRNN(128, return_sequences=True)(inputs)
    rnn2 = tf.keras.layers.SimpleRNN(128)(rnn1)
    x = tf.keras.layers.Dropout(0.20)(rnn2)
    x = tf.keras.layers.Dense(datasize)(x)
    outputs = tf.keras.layers.Activation('relu')(x)
    # outputs = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return tf.keras.Model(inputs, outputs)

def lstm_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))
    lstm_layer1 = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    lstm_layer2 = tf.keras.layers.LSTM(128)(lstm_layer1)
    dropout = tf.keras.layers.Dropout(0.2)(lstm_layer2)
    dense = tf.keras.layers.Dense(datasize)(dropout)
    relu = tf.keras.layers.Activation('relu')(dense)
    return tf.keras.Model(inputs=inputs, outputs=relu)
    # leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)(dense)
    # return tf.keras.Model(inputs=inputs, outputs=leaky_relu)
    
    

def tcn_model(window_size, datasize):
    inputs = tf.keras.Input(shape=(window_size, datasize))
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(datasize)(x)
    outputs = tf.keras.layers.Activation('relu')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def transformer_model(window_size, datasize, num_heads=5, ff_dim=128, num_layers=3, dropout_rate=0):

    inputs = tf.keras.Input(shape=(window_size, datasize))

    # Positional encoding
    def positional_encoding(length, depth):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

    pos_encoding = positional_encoding(window_size, datasize)
    x = inputs + pos_encoding[tf.newaxis, :, :]  # (batch, seq, dim)

    # 💡 Causal mask : lower triangular (1s on and below the diagonal)
    def create_causal_mask(seq_len):
        return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    causal_mask = create_causal_mask(window_size)  # shape: (window_size, window_size)

    for _ in range(num_layers):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=datasize // num_heads,
        )(x, x, attention_mask=causal_mask)
        
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ff_output = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(datasize),
            tf.keras.layers.Dropout(dropout_rate)
        ])(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        


    # Prédiction basée sur la dernière position de la séquence (t5 → t6)
    outputs = tf.keras.layers.Dense(datasize)(x[:, -1, :])
    outputs = tf.keras.layers.LeakyReLU(alpha=0.01)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Création et compilation du modèle
def build_model(model_type, window_size, datasize):
    if model_type == "rnn":
        model = rnn_model(window_size, datasize)
        batch_size = 128
    elif model_type == "lstm":
        model = lstm_model(window_size, datasize)
        batch_size = 64
    elif model_type == "tcn":
        model = tcn_model(window_size, datasize)
        batch_size = 64
    elif model_type == "transformer":
        pred_len = 3 
        model = transformer_model(window_size, datasize)
        batch_size = 32
    else:
        raise ValueError("Model type must be 'rnn', 'lstm', 'tcn', or 'transformer'")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mae')
    model.summary()
    return model, batch_size

# Entraînement du modèle
def train_model(model, X_train, Y_train, X_validation, Y_validation, batch_size, model_type):
    hist = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=300,
        validation_data=(X_validation, Y_validation)
    )

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(14, 9))
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss ({model_type.upper()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    training_loss_path = os.path.join(base_output_dir, f"training_validation_loss_{model_type}.png")
    plt.tight_layout()
    plt.savefig(training_loss_path, bbox_inches="tight")
    # plt.savefig(training_loss_path)
    print(f"Plot of training/validation loss saved to {training_loss_path}")
    plt.close()

# Prédiction et dénormalisation
def predict_and_denormalize(model, sequences_array, sequences_sums_array, seq_dirs_test, sequence_dirs, output_size, window_size, model_type):
    predictions_all = []
    denormalized_predictions_all = []
    denormalized_y_test_all = []

    for seq_idx in range(len(seq_dirs_test)):
        sequence_dir = seq_dirs_test[seq_idx]
        seq_full = sequences_array[sequence_dirs.index(sequence_dir)]
        sums_full = sequences_sums_array[sequence_dirs.index(sequence_dir)]
        
        if len(seq_full) < window_size + output_size:
            print(f"Séquence {sequence_dir} trop courte pour prédire {output_size} pas de temps.")
            continue
        
        current_window = seq_full[:window_size].copy()
        predictions = []
        
        for t in range(window_size, window_size + output_size):
            current_window_input = current_window[np.newaxis, ...]
            pred = model.predict(current_window_input, verbose=0)
            predictions.append(pred[0])
            current_window = np.vstack((current_window[1:], pred[0]))
        
        predictions = np.array(predictions)
        predictions_all.append(predictions)
        
        actual_sums = sums_full[window_size:window_size + output_size]
        denorm_preds = predictions * actual_sums[:, np.newaxis]
        denorm_y_test = seq_full[window_size:window_size + output_size] * actual_sums[:, np.newaxis]
        denorm_preds = np.maximum(denorm_preds, 0)
        
        denormalized_predictions_all.append(denorm_preds)
        denormalized_y_test_all.append(denorm_y_test)

    return np.array(denormalized_predictions_all), np.array(denormalized_y_test_all), predictions_all

# Sauvegarde des prédictions et génération des figures
def save_predictions_and_figures(denormalized_predictions_all, denormalized_y_test_all, seq_dirs_test, sequence_dirs, file_names, output_size, datasize, output_dir, output_dir2, y_test_dir, model_type):
    predictions_txt_dir = os.path.join(output_dir, f"predictions_txt_{model_type}")
    os.makedirs(predictions_txt_dir, exist_ok=True)

    x_values = np.linspace(0, xmax, datasize)
    for seq_idx in range(len(seq_dirs_test)):
        sequence_dir = seq_dirs_test[seq_idx]
        sequence_file_names = file_names[sequence_dirs.index(sequence_dir)][window_size:window_size + output_size]
        
        if seq_idx >= len(denormalized_predictions_all):
            continue
        
        max_y_value = np.max([
            np.max(denormalized_y_test_all[seq_idx]),
            np.max(denormalized_predictions_all[seq_idx])
        ])
        y_limit = max_y_value * 1.1
        
        sequence_output_dir = os.path.join(output_dir2, f"{sequence_dir}_{model_type}")
        os.makedirs(sequence_output_dir, exist_ok=True)
        
        sequence_y_test_dir = os.path.join(y_test_dir, f"{sequence_dir}_{model_type}")
        os.makedirs(sequence_y_test_dir, exist_ok=True)
        
        for time_step in range(output_size):
            original_file_name = sequence_file_names[time_step]
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, denormalized_y_test_all[seq_idx, time_step, :],
                     label=f"Ground truth")
            plt.plot(x_values, denormalized_predictions_all[seq_idx, time_step, :],
                     label=f"Prediction {model_type.upper()}", linestyle='--')
            
            plt.ylim(0, y_limit)
            plt.xlabel("Grain Size (mm)")
            # plt.ylabel("Grain Surface (mm²)")
            plt.ylabel("Frequency")
            plt.title(f"Time Step {time_step + window_size+1}")
            plt.legend()
            
            figure_path = os.path.join(sequence_output_dir, f"{time_step + window_size}.png")
            plt.savefig(figure_path)
            plt.close()
            
            pred_file_path = os.path.join(predictions_txt_dir, f"{sequence_dir}_{original_file_name}.txt")
            with open(pred_file_path, 'w') as pred_file:
                pred_file.write(f"Y_test dénormalisé ({original_file_name}):\n")
                pred_file.write('\n'.join(map(str, denormalized_y_test_all[seq_idx, time_step, :])))
                pred_file.write(f"\n\nPrédiction dénormalisée {model_type.upper()}:\n")
                pred_file.write('\n'.join(map(str, denormalized_predictions_all[seq_idx, time_step, :])))
            
            y_test_file_path = os.path.join(sequence_y_test_dir, f"{original_file_name}.txt")
            with open(y_test_file_path, 'w') as y_test_file:
                y_test_file.write(f"Y_test dénormalisé ({original_file_name}):\n")
                y_test_file.write('\n'.join(map(str, denormalized_y_test_all[seq_idx, time_step, :])))
    
    return predictions_txt_dir

# Calcul des erreurs
def calculate_errors(denormalized_predictions_all, denormalized_y_test_all, seq_dirs_test, model_type):
    mae = np.mean(np.abs(denormalized_predictions_all - denormalized_y_test_all))
    mse = np.mean((denormalized_predictions_all - denormalized_y_test_all) ** 2)
    rmse = np.sqrt(mse)
    mask = denormalized_y_test_all != 0
    mre = np.mean(np.abs((denormalized_y_test_all[mask] - denormalized_predictions_all[mask]) / denormalized_y_test_all[mask])) * 100

    print(f"\nMean Absolute Error (MAE) {model_type.upper()}: {mae:.4f}")
    print(f"Mean Squared Error (MSE) {model_type.upper()}: {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE) {model_type.upper()}: {rmse:.4f}")
    print(f"Mean Relative Error (MRE) {model_type.upper()} en pourcentage: {mre:.2f}%")

    print(f"\nSéquences dans X_test et Y_test ({model_type.upper()}):")
    for i, seq_name in enumerate(seq_dirs_test):
        print(f"Échantillon de test {i} : {seq_name}")

    mae_per_sequence = []
    mse_per_sequence = []
    rmse_per_sequence = []
    mre_per_sequence = []

    print(f"\nErrors for Each Test Sequence ({model_type.upper()}):")
    for seq_idx in range(len(seq_dirs_test)):
        sequence_name = seq_dirs_test[seq_idx]
        if seq_idx >= len(denormalized_predictions_all):
            print(f"\nSequence: {sequence_name} (non prédite, séquence trop courte)")
            continue
        
        y_true_seq = denormalized_y_test_all[seq_idx]
        y_pred_seq = denormalized_predictions_all[seq_idx]
        
        mae = np.mean(np.abs(y_true_seq - y_pred_seq))
        mae_per_sequence.append(mae)
        
        mse = np.mean((y_true_seq - y_pred_seq) ** 2)
        mse_per_sequence.append(mse)
        
        rmse = np.sqrt(mse)
        rmse_per_sequence.append(rmse)
        
        mask = (y_true_seq > 0) & (y_pred_seq >= 0)
        if np.any(mask):
            relative_errors = np.abs((y_true_seq[mask] - y_pred_seq[mask]) / y_true_seq[mask])
            mre = np.sum(relative_errors) / y_true_seq.size * 100
        else:
            mre = 0
        mre_per_sequence.append(mre)
        
        print(f"\nSequence: {sequence_name}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  Mean Relative Error (MRE): {mre:.2f}%")

    mean_mae = np.mean(mae_per_sequence) if mae_per_sequence else 0
    mean_mse = np.mean(mse_per_sequence) if mse_per_sequence else 0
    mean_rmse = np.mean(rmse_per_sequence) if rmse_per_sequence else 0
    mean_mre = np.mean(mre_per_sequence) if mre_per_sequence else 0

    print(f"\nMean Errors Across All Test Sequences ({model_type.upper()}):")
    print(f"  Mean MAE: {mean_mae:.4f}")
    print(f"  Mean MSE: {mean_mse:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}")
    print(f"  Mean MRE: {mean_mre:.2f}%")

    return mean_mae, mean_mse, mean_rmse, mean_mre

# Nouvelle fonction pour tester une nouvelle séquence
def test_new_sequence(model, new_sequence_dir, output_size, window_size, datasize, model_type, base_output_dir):
    # Charger et normaliser la nouvelle séquence
    sequence = []
    sequence_sums = []
    sequence_files = []
    
    normalized_new_sequence_dir = os.path.join(base_output_dir, f"normalisation_new_sequence_{model_type}")
    reset_directory(normalized_new_sequence_dir)
    
    for file_name in sorted(os.listdir(new_sequence_dir), key=numerical_sort):
        file_path = os.path.join(new_sequence_dir, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r') as file:
                next(file)
                file_content = file.read().strip().split()
                file_content_float = [float(val) for val in file_content]
                
                sum_value = sum(file_content_float)
                sequence_sums.append(sum_value)
                
                # if sum_value != 0:
                #     normalized_values = [val / sum_value for val in file_content_float]
                # else:
                #     normalized_values = [0 for _ in file_content_float]
                if sum_value != 0:
                    normalized_values = [round(val / sum_value, 16) for val in file_content_float]
                else:
                    normalized_values = [0.0 for _ in file_content_float]


                
                sequence.append(normalized_values)
                sequence_files.append(file_name)
                
                normalized_file_path = os.path.join(normalized_new_sequence_dir, file_name)
                # with open(normalized_file_path, 'w') as norm_file:
                #     norm_file.write('\n'.join(map(str, normalized_values)))
    
                with open(normalized_file_path, 'w') as norm_file:
                    norm_file.write('\n'.join(f"{val:.8f}" for val in normalized_values))
    
    sequence_array = np.array(sequence)
    sums_array = np.array(sequence_sums)
    
    if len(sequence_array) < window_size + output_size:
        print(f"Nouvelle séquence trop courte pour prédire {output_size} pas de temps.")
        return None, None, None, None, None, None
    
    # Prédire
    current_window = sequence_array[:window_size].copy()
    predictions = []
    
    for t in range(window_size, window_size + output_size):
        current_window_input = current_window[np.newaxis, ...]
        pred = model.predict(current_window_input, verbose=0)
        predictions.append(pred[0])
        current_window = np.vstack((current_window[1:], pred[0]))
    
    predictions = np.array(predictions)
    actual_sums = sums_array[window_size:window_size + output_size]
    denorm_preds = predictions * actual_sums[:, np.newaxis]
    denorm_y_test = sequence_array[window_size:window_size + output_size] * actual_sums[:, np.newaxis]
    denorm_preds = np.maximum(denorm_preds, 0)
    
    # Sauvegarder les prédictions et générer les figures
    new_sequence_output_dir = os.path.join(base_output_dir, f"predictions_new_sequence_{model_type}")
    new_sequence_figures_dir = os.path.join(base_output_dir, f"figures_new_sequence_{model_type}")
    new_sequence_y_test_dir = os.path.join(base_output_dir, f"ytest_new_sequence_{model_type}")
    
    os.makedirs(new_sequence_output_dir, exist_ok=True)
    os.makedirs(new_sequence_figures_dir, exist_ok=True)
    os.makedirs(new_sequence_y_test_dir, exist_ok=True)
    
    x_values = np.linspace(0, xmax, datasize)
    max_y_value = np.max([np.max(denorm_y_test), np.max(denorm_preds)])
    y_limit = max_y_value * 1.1
    
    for time_step in range(output_size):
        original_file_name = sequence_files[window_size + time_step]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, denorm_y_test[time_step, :], label=f"Y_test réel", color='black', linewidth=2)  # Thicker line for real data
        plt.plot(x_values, denorm_preds[time_step, :], label=f"Prédiction {model_type.upper()}", linestyle='--', color='darkgreen', linewidth=2)  # Thicker dark green line for prediction
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, denorm_y_test[time_step, :], label=f"Y_test réel")
        # plt.plot(x_values, denorm_preds[time_step, :], label=f"Prédiction {model_type.upper()}", linestyle='--')
        
        plt.ylim(0, y_limit)
        plt.xlabel("Grain Size(mm)")
        plt.ylabel("Frequency")
        plt.title(f"Time Step {time_step + window_size + 1}")
        plt.legend()
        
        figure_path = os.path.join(new_sequence_figures_dir, f"{time_step + window_size}.png")
        plt.savefig(figure_path)
        plt.close()
        
        pred_file_path = os.path.join(new_sequence_output_dir, f"new_sequence_{original_file_name}.txt")
        with open(pred_file_path, 'w') as pred_file:
            pred_file.write(f"Y_test dénormalisé ({original_file_name}):\n")
            pred_file.write('\n'.join(map(str, denorm_y_test[time_step, :])))
            pred_file.write(f"\n\nPrédiction dénormalisée {model_type.upper()}:\n")
            pred_file.write('\n'.join(map(str, denorm_preds[time_step, :])))
        
        y_test_file_path = os.path.join(new_sequence_y_test_dir, f"{original_file_name}.txt")
        with open(y_test_file_path, 'w') as y_test_file:
            y_test_file.write(f"Y_test dénormalisé ({original_file_name}):\n")
            y_test_file.write('\n'.join(map(str, denorm_y_test[time_step, :])))
    
    # Calculer les erreurs
    mae = np.mean(np.abs(denorm_preds - denorm_y_test))
    mse = np.mean((denorm_preds - denorm_y_test) ** 2)
    rmse = np.sqrt(mse)
    mask = (denorm_y_test > 0) & (denorm_preds >= 0)
    if np.any(mask):
        relative_errors = np.abs((denorm_y_test[mask] - denorm_preds[mask]) / denorm_y_test[mask])
        mre = np.sum(relative_errors) / denorm_y_test.size * 100
    else:
        mre = 0
    
    print(f"\nErrors for New Sequence ({model_type.upper()}):")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Relative Error (MRE): {mre:.2f}%")
    
    return denorm_preds, denorm_y_test, mae, mse, rmse, mre


def generate_combined_figures(all_predictions,
                              all_y_test,
                              seq_dirs_test,
                              sequence_dirs,
                              file_names,
                              output_size,
                              datasize,
                              base_output_dir,
                              model_types,
                              yscale_mode="per_timestep",   # "per_timestep" | "per_sequence" | "fixed"
                              fixed_ylim=None):
    """
    yscale_mode:
        - "per_timestep"  : y_max recalculé pour chaque time_step ( ce que tu demandes)
        - "per_sequence"  : y_max unique pour toute la séquence (comportement initial)
        - "fixed"         : y_max fixé à `fixed_ylim` (float > 0), pratique pour comparer visuellement
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    figures_all_dir = os.path.join(base_output_dir, "figures_all")
    os.makedirs(figures_all_dir, exist_ok=True)

    # # Espacement visuel entre groupes
    # group_gap = 0.01
    # x_groups = np.arange(datasize) * group_gap  # positions espacées

    # total_bars = len(model_types) + 1  # +1 pour Y_test
    # bar_width = 0.0015
    # --- Axe X fidèle à 0→0.11, bords/centres/largeur de bin ---
    x_min, x_max = 0.0, xmax
    bin_edges = np.linspace(x_min, x_max, datasize + 1)        # 31 bords pour 30 bins
    x_values = (bin_edges[:-1] + bin_edges[1:]) / 2.0          # centres (30)
    bin_width = (x_max - x_min) / datasize

    x_groups = x_values                                        # on utilise les centres
    total_bars = len(model_types) + 1                          # +1 pour y_test
    bar_width  = bin_width * 0.8 / total_bars                  # 80% du bin réparti
    edge_pad   = bin_width * 0.1                               # petite marge aux bords

    colors = {
        'y_test': '#444444',
        'rnn': 'green',
        'lstm': 'red',
        'tcn': 'orange',
        'transformer': 'blue',
    }

    custom_labels = {
        'rnn': 'RNN prediction',
        'lstm': 'LSTM prediction',
        'tcn': 'TCN prediction',
        'transformer': 'TRANSFORMER prediction',
        'y_test': 'Ground truth',
    }

    for seq_idx in range(len(seq_dirs_test)):
        sequence_dir = seq_dirs_test[seq_idx]
        sequence_file_names = file_names[sequence_dirs.index(sequence_dir)][window_size:window_size + output_size]
        first_model = model_types[0]

        if seq_idx >= len(all_y_test[first_model]):
            continue

        # --- Pré-calcul du y_max par séquence si besoin ---
        if yscale_mode == "per_sequence":
            seq_max = 0.0
            for t in range(output_size):
                # y_test
                seq_max = max(seq_max, float(np.max(all_y_test[first_model][seq_idx, t, :])))
                # predictions
                for model_type in model_types:
                    if model_type in all_predictions:
                        seq_max = max(seq_max, float(np.max(all_predictions[model_type][seq_idx, t, :])))
            seq_y_limit = seq_max * 1.1 if seq_max > 0 else 1.0

        # --- Dossier de sortie pour cette séquence ---
        sequence_output_dir = os.path.join(figures_all_dir, sequence_dir)
        os.makedirs(sequence_output_dir, exist_ok=True)

        # --- Boucle sur les time steps ---
        for time_step in range(output_size):
            original_file_name = sequence_file_names[time_step]

            # Calcul du y_limit selon le mode demandé
            if yscale_mode == "per_timestep":
                # y_max basé uniquement sur ce time_step
                ts_max = float(np.max(all_y_test[first_model][seq_idx, time_step, :]))
                for model_type in model_types:
                    if model_type in all_predictions:
                        ts_max = max(ts_max, float(np.max(all_predictions[model_type][seq_idx, time_step, :])))
                y_limit = ts_max * 1.1 if ts_max > 0 else 1.0

            elif yscale_mode == "per_sequence":
                y_limit = seq_y_limit

            elif yscale_mode == "fixed":
                if fixed_ylim is None or fixed_ylim <= 0:
                    raise ValueError("Avec yscale_mode='fixed', fournis un fixed_ylim > 0.")
                y_limit = float(fixed_ylim)

            else:
                raise ValueError("yscale_mode doit être 'per_timestep', 'per_sequence' ou 'fixed'.")

            # --- Plot ---
            plt.figure(figsize=(16, 10))
            for bin_idx, x_val in enumerate(x_groups):
                for model_offset, model_type in enumerate(['y_test'] + model_types):
                    if model_type == 'y_test':
                        y = all_y_test[first_model][seq_idx, time_step, bin_idx]
                    else:
                        y = all_predictions[model_type][seq_idx, time_step, bin_idx]

                    # offset = (model_offset - total_bars / 2) * bar_width + bar_width / 2
                    # x_pos = x_val + offset

                    offset = (model_offset - (total_bars - 1) / 2.0) * bar_width
                    x_pos  = x_val + offset

                    plt.bar(
                        x_pos, y, width=bar_width,
                        color=colors.get(model_type, 'gray'),
                        label=custom_labels.get(model_type, model_type.upper()) if bin_idx == 0 else ""
                    )
            plt.xlim(x_min - edge_pad, x_max + edge_pad)
            plt.margins(x=0)  # pas de marge auto supplémentaire
            plt.ylim(0, y_limit)
            plt.xlabel("Grain Size (mm)", fontsize=19)
            plt.ylabel("Frequency", fontsize=19)
            plt.title(f"Time Step {time_step + window_size + 1}")
            plt.xticks(x_groups[::2], [f"{x:.3f}" for x in x_groups[::2]], rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()

            figure_path = os.path.join(sequence_output_dir, f"{time_step + window_size}.png")
            plt.savefig(figure_path)
            plt.close()

def test_new_sequences(model,
                       sequences_root_dir,
                       output_size,
                       window_size,
                       datasize,
                       model_type,
                       base_output_dir):
    """
    Teste un lot de nouvelles séquences situées chacune dans un sous-dossier de `sequences_root_dir`.

    Pour chaque séquence:
      - normalise et sauvegarde la version normalisée
      - exécute le modèle en fenêtre glissante auto-régressive sur `output_size`
      - dénormalise avec les sommes réelles
      - sauvegarde .txt, figures et y_test
      - calcule MAE, MSE, RMSE, MRE par séquence

    Retourne:
      - results_per_sequence: dict {seq_name: {"MAE":..., "MSE":..., "RMSE":..., "MRE":..., "n_steps": int}}
      - global_metrics: dict {"MAE":..., "MSE":..., "RMSE":..., "MRE":...} (moyennes pondérées par le nb d’items)
    """

    # Dossiers de sortie (groupés)
    norm_root = os.path.join(base_output_dir, f"normalisation_new_sequences_{model_type}")
    preds_root = os.path.join(base_output_dir, f"predictions_new_sequences_{model_type}")
    figs_root  = os.path.join(base_output_dir, f"figures_new_sequences_{model_type}")
    ytest_root = os.path.join(base_output_dir, f"ytest_new_sequences_{model_type}")

    reset_directory(norm_root)
    os.makedirs(preds_root, exist_ok=True)
    os.makedirs(figs_root, exist_ok=True)
    os.makedirs(ytest_root, exist_ok=True)

    # Accumulateurs globaux pour les métriques
    abs_err_sum = 0.0
    sq_err_sum  = 0.0
    rel_err_sum = 0.0
    n_items     = 0         # nb total d'éléments (time_steps * datasize) utilisés pour MAE/MSE/RMSE
    n_items_rel = 0         # nb total d'éléments valides pour MRE

    results_per_sequence = {}

    # Axe X pour les figures
    x_values = np.linspace(0, xmax, datasize)

    # Parcours des sous-dossiers (une séquence par dossier)
    for seq_name in sorted(os.listdir(sequences_root_dir), key=numerical_sort):
        seq_path = os.path.join(sequences_root_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue

        # ---------- 1) Charger & normaliser la séquence ----------
        sequence = []
        sequence_sums = []
        sequence_files = []

        norm_seq_dir = os.path.join(norm_root, seq_name)
        os.makedirs(norm_seq_dir, exist_ok=True)

        for file_name in sorted(os.listdir(seq_path), key=numerical_sort):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(seq_path, file_name)
            with open(file_path, 'r') as f:
                next(f)  # saute la première ligne
                vals = f.read().strip().split()
                vals = [float(v) for v in vals]
            s = sum(vals)
            sequence_sums.append(s)
            # if s != 0:
            #     normalized = [v / s for v in vals]
            # else:
            #     normalized = [0.0 for _ in vals]
                
            if s != 0:
                normalized = [round(v / s, 16) for v in vals]
            else:
                normalized = [0.0 for _ in vals]
            
            sequence.append(normalized)
            sequence_files.append(file_name)

            # Sauvegarde normalisée
            with open(os.path.join(norm_seq_dir, file_name), "w") as nf:
               nf.write("\n".join(f"{val:.8f}" for val in normalized))
            # with open(os.path.join(norm_seq_dir, file_name), "w") as nf:
            #     nf.write("\n".join(map(str, normalized)))

        if len(sequence) == 0:
            print(f"[{seq_name}] Aucun fichier .txt trouvé, ignoré.")
            continue

        sequence = np.array(sequence, dtype=float)
        sums_arr = np.array(sequence_sums, dtype=float)

        if len(sequence) < window_size + output_size:
            print(f"[{seq_name}] Séquence trop courte pour prédire {output_size} pas de temps, ignorée.")
            continue

        # ---------- 2) Prédire en auto-régressif ----------
        current_window = sequence[:window_size].copy()
        preds = []
        for t in range(window_size, window_size + output_size):
            inp = current_window[np.newaxis, ...]  # (1, window, datasize)
            pred = model.predict(inp, verbose=0)[0]  # (datasize,)
            preds.append(pred)
            # Glissement de fenêtre: drop premier, append pred
            current_window = np.vstack((current_window[1:], pred))

        preds = np.array(preds)  # (output_size, datasize)

        # ---------- 3) Dénormaliser ----------
        actual_sums = sums_arr[window_size:window_size + output_size]                     # (output_size,)
        denorm_preds = preds * actual_sums[:, np.newaxis]                                 # (output_size, datasize)
        denorm_y    = sequence[window_size:window_size + output_size] * actual_sums[:, np.newaxis]
        denorm_preds = np.maximum(denorm_preds, 0)

        # ---------- 4) Sauvegardes ----------
        seq_fig_dir  = os.path.join(figs_root,  f"{seq_name}_{model_type}")
        seq_pred_dir = os.path.join(preds_root, f"{seq_name}_{model_type}")
        seq_y_dir    = os.path.join(ytest_root, f"{seq_name}_{model_type}")
        os.makedirs(seq_fig_dir, exist_ok=True)
        os.makedirs(seq_pred_dir, exist_ok=True)
        os.makedirs(seq_y_dir, exist_ok=True)

        max_y_value = float(np.max([np.max(denorm_y), np.max(denorm_preds)]))
        y_limit = max_y_value * 1.1 if max_y_value > 0 else 1.0

        for t in range(output_size):
            original_file = sequence_files[window_size + t]

            # Figure
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, denorm_y[t, :], label="Y_test réel", color='black', linewidth=2)
            plt.plot(x_values, denorm_preds[t, :], label=f"Prédiction {model_type.upper()}",
                     linestyle='--', color='darkgreen', linewidth=2)
            plt.ylim(0, y_limit)
            plt.xlabel("Grain Size(mm)")
            plt.ylabel("Frequency")
            plt.title(f"{seq_name} — Time Step {t + window_size + 1}")
            plt.legend()
            plt.savefig(os.path.join(seq_fig_dir, f"{t + window_size}.png"))
            plt.close()

            # .txt (prédictions + y_test)
            with open(os.path.join(seq_pred_dir, f"{original_file}.txt"), "w") as pf:
                pf.write(f"Y_test dénormalisé ({original_file}):\n")
                pf.write("\n".join(map(str, denorm_y[t, :])))
                pf.write(f"\n\nPrédiction dénormalisée {model_type.upper()}:\n")
                pf.write("\n".join(map(str, denorm_preds[t, :])))

            with open(os.path.join(seq_y_dir, f"{original_file}.txt"), "w") as yf:
                yf.write(f"Y_test dénormalisé ({original_file}):\n")
                yf.write("\n".join(map(str, denorm_y[t, :])))

        # ---------- 5) Erreurs par séquence ----------
        diff = denorm_preds - denorm_y
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff**2))
        rmse = float(np.sqrt(mse))

        mask = (denorm_y > 0) & (denorm_preds >= 0)
        if np.any(mask):
            rel = np.abs((denorm_y[mask] - denorm_preds[mask]) / denorm_y[mask])
            # MRE ici = moyenne sur toute la grille (comme ton implémentation)
            mre = float(np.sum(rel) / denorm_y.size * 100)
            rel_count = int(np.count_nonzero(mask))
        else:
            mre = 0.0
            rel_count = 0

        results_per_sequence[seq_name] = {
            "MAE": mae, "MSE": mse, "RMSE": rmse, "MRE": mre,
            "n_steps": int(output_size)
        }

        # Accumulation globale (pondérée par nb d'éléments)
        n = denorm_y.size
        abs_err_sum += float(np.sum(np.abs(diff)))
        sq_err_sum  += float(np.sum(diff**2))
        n_items     += n
        rel_err_sum += float(np.sum(rel)) if rel_count > 0 else 0.0
        n_items_rel += denorm_y.size     if rel_count > 0 else 0

        # Log
        print(f"\n[{seq_name}] Errors ({model_type.upper()}):")
        print(f"  MAE : {mae:.4f}")
        print(f"  MSE : {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MRE : {mre:.2f}%")

    # ---------- 6) Métriques globales ----------
    if n_items > 0:
        global_mae  = abs_err_sum / n_items
        global_mse  = sq_err_sum  / n_items
        global_rmse = float(np.sqrt(global_mse))
    else:
        global_mae = global_mse = global_rmse = 0.0

    if n_items_rel > 0:
        # même définition que ci-dessus: somme des erreurs relatives / total éléments
        global_mre = (rel_err_sum / n_items_rel) * 100.0
    else:
        global_mre = 0.0

    global_metrics = {
        "MAE": global_mae,
        "MSE": global_mse,
        "RMSE": global_rmse,
        "MRE": global_mre
    }

    print("\n=== Global errors across all NEW sequences ===")
    print(f"  MAE : {global_mae:.4f}")
    print(f"  MSE : {global_mse:.4f}")
    print(f"  RMSE: {global_rmse:.4f}")
    print(f"  MRE : {global_mre:.2f}%")

    return results_per_sequence, global_metrics



def main():
    # 1) Valider la liste de modèles demandés
    for model in model_types:
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {model} non valide. Choisissez parmi {AVAILABLE_MODELS}")

    # 2) Charger + normaliser les données UNE fois
    #    (la fonction écrit normalisation2/ -> on le supprime juste après)
    normalized_output_dir = os.path.join(base_output_dir, "normalisation2")
    sequences_array, sequences_sums_array, file_names, sequence_dirs = load_and_normalize_data(normalized_output_dir)

    # On supprime les fichiers normalisés pour ne garder AUCUNE autre sortie disque
    if os.path.exists(normalized_output_dir):
        shutil.rmtree(normalized_output_dir, ignore_errors=True)

    # 3) Construire les datasets (en mémoire — n’écrit rien)
    (X_train, Y_train, Y_sums_train,
     X_validation, Y_validation, Y_sums_validation,
     X_test, Y_test, Y_sums_test,
     seq_dirs_train, seq_dirs_validation, seq_dirs_test) = prepare_datasets(
        sequences_array, sequences_sums_array, sequence_dirs
    )

    # Conteneurs pour les figures combinées (en mémoire)
    all_predictions = {}
    all_y_test = {}

    # 4) Boucle sur les modèles
    for model_type in model_types:
        print(f"\n=== Exécution du modèle {model_type.upper()} ===\n")

        # a) Construire/Compiler
        model, batch_size = build_model(model_type, window_size, datasize)

        # b) Entraîner -> génère UNIQUEMENT le PNG 'training_validation_loss_{model}.png'
        train_model(model, X_train, Y_train, X_validation, Y_validation, batch_size, model_type)

        # c) Prédire sur le split test (mémoire uniquement)
        denorm_preds_all, denorm_y_all, _ = predict_and_denormalize(
            model=model,
            sequences_array=sequences_array,
            sequences_sums_array=sequences_sums_array,
            seq_dirs_test=seq_dirs_test,
            sequence_dirs=sequence_dirs,
            output_size=output_size,
            window_size=window_size,
            model_type=model_type
        )

        # d) Stocker pour figures combinées (aucune écriture ici)
        all_predictions[model_type] = denorm_preds_all
        all_y_test[model_type] = denorm_y_all

        # (Optionnel) Afficher juste les métriques en console (pas de fichiers)
        _ = calculate_errors(denorm_preds_all, denorm_y_all, seq_dirs_test, model_type)


      # sauvegarde du modèle
        model_save_dir = os.path.join(base_output_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        model_path = os.path.join(model_save_dir, f"{model_type}_model.keras")
        model.save(model_path)
        print(f"Modèle {model_type.upper()} sauvegardé dans : {model_path}")
        
        # Nettoyer le graphe TF
        tf.keras.backend.clear_session()
        print(f"=== Fin du modèle {model_type.upper()} ===\n")

    # 5) Générer les figures combinées multi-modèles si >1 modèle
    if len(model_types) > 1:
        generate_combined_figures(
            all_predictions=all_predictions,
            all_y_test=all_y_test,
            seq_dirs_test=seq_dirs_test,
            sequence_dirs=sequence_dirs,
            file_names=file_names,
            output_size=output_size,
            datasize=datasize,
            base_output_dir=base_output_dir,
            model_types=model_types
        )
    else:
        print(f"Seul un modèle ({model_types[0].upper()}) a été exécuté : pas de figures combinées.")

    # 6) Petits récap’ en console (aucune écriture disque)
    print("\n=== Récapitulatif terminé (aucune sortie autre que loss + figures_all si >1 modèle) ===")


if __name__ == "__main__":
    main()
