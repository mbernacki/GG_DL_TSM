# This script evaluates the Burke–Turnbull grain-growth law using reference and model-predicted distributions.
# It analyzes R_bar(t)^2 - R_bar(0)^2, its slope K, linearity, and prediction accuracy.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import shutil
import random
import gc
import csv

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# ============================================================
# FIGURE STYLE
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

base_output_dir = "/media/admin-eyounes/T7/1-Article/github_code/SN_output/1heure"
new_sequences_root = "/media/admin-eyounes/T7/1-Article/github_code/test_sequences"

# base_output_dir = (
#     "/home/admin-eyounes/Desktop/29_janv/1/"
#     "données+code2/article/SN_output/1heure"
# )

# new_sequences_root = (
#     "/home/admin-eyounes/Desktop/29_janv/1/"
#     "données+code2/article/test_sequences"
# )


# Number of grain-size bins
datasize = 30

# Number of known time steps provided to the model
window_size = 5

# Number of future time steps to predict
output_size = 55


AVAILABLE_MODELS = [
    "transformer",
    "rnn",
    "lstm",
    "tcn",
]

model_types = [
    "rnn",
    "lstm",
    "tcn",
    "transformer",
]


# Time axis
USE_MINUTES = False

# Duration of one time step when USE_MINUTES = True
DT_MIN = 1.0


# ============================================================
# GRAIN-SIZE BINS
# ============================================================

# Grain sizes are distributed between 0 and 0.17 mm.
#
# IMPORTANT:
#   - if these values represent radii, retain R_bar;
#   - if they represent diameters, replace the
#     R_bar labels with D_bar in the figures and CSV files.

bins = np.linspace(
    0.0,
    0.17,
    datasize + 1
)

BIN_CENTERS = 0.5 * (
    bins[:-1] + bins[1:]
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

out_root = os.path.join(
    base_output_dir,
    "SN_BURKE_TURNBULL_R2_MINUS_R0_2"
)


# ============================================================
# COLORS AND LABELS
# ============================================================

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


# ============================================================
# GENERAL UTILITY FUNCTIONS
# ============================================================

def reset_directory(directory: str):
    """
    Removes the directory if it exists and then recreates it.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(
        directory,
        exist_ok=True
    )


def safe_makedirs(path: str):
    """
    Creates the directory if it does not exist.
    """
    os.makedirs(
        path,
        exist_ok=True
    )


def numerical_sort(value):
    """
    Natural sorting:
      file2.txt appears before file10.txt.
    """
    parts = re.split(
        r"(\d+)",
        value
    )

    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


# ============================================================
# DATA LOADING
# ============================================================

def load_one_sequence_in_memory(sequence_dir: str):
    """
    Loads all .txt files belonging to one sequence.

    Returns
    -------
    seq_norm : ndarray, shape (T, datasize)
        Normalized grain-size distributions.

    sums : ndarray, shape (T,)
        Original distribution sums before normalization.

    files : list[str]
        List of loaded files.
    """

    seq = []
    sums = []
    files = []

    for file_name in sorted(
        os.listdir(sequence_dir),
        key=numerical_sort
    ):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(
            sequence_dir,
            file_name
        )

        with open(file_path, "r") as file:
            # Skip the first line
            next(file)

            vals = file.read().strip().split()
            vals = [
                float(value)
                for value in vals
            ]

        if len(vals) != datasize:
            raise ValueError(
                f"Fichier {file_path}: "
                f"attendu {datasize} valeurs, "
                f"reçu {len(vals)}"
            )

        distribution_sum = float(
            sum(vals)
        )

        sums.append(
            distribution_sum
        )

        if distribution_sum != 0.0:
            normalized_distribution = [
                round(
                    value / distribution_sum,
                    16
                )
                for value in vals
            ]
        else:
            normalized_distribution = [
                0.0
                for _ in vals
            ]

        seq.append(
            normalized_distribution
        )

        files.append(
            file_name
        )

    if len(seq) == 0:
        return None, None, None

    return (
        np.asarray(seq, dtype=float),
        np.asarray(sums, dtype=float),
        files
    )


# ============================================================
# AUTOREGRESSIVE PREDICTION
# ============================================================

def autoregressive_predict_norm(
    model,
    seq_norm,
    window_size,
    output_size
):
    """
    Performs autoregressive predictions on normalized distributions.

    The model receives the first window_size time steps. Each prediction
    is then fed back into the window to generate the next time step.

    Returns
    -------
    preds_norm : ndarray, shape (output_size, datasize)
    """

    current_window = (
        seq_norm[:window_size].copy()
    )

    predictions = []

    for _ in range(output_size):

        model_input = current_window[
            np.newaxis,
            ...
        ]

        prediction = model.predict(
            model_input,
            verbose=0
        )[0]

        # Prevent negative values
        prediction = np.maximum(
            prediction,
            0.0
        )

        # Renormalize the distribution
        prediction_sum = (
            np.sum(prediction) + 1e-8
        )

        prediction_normalized = (
            prediction / prediction_sum
        )

        predictions.append(
            prediction_normalized
        )

        # Shift the window and append the new prediction
        current_window = np.vstack((
            current_window[1:],
            prediction_normalized
        ))

    return np.asarray(
        predictions,
        dtype=float
    )


# ============================================================
# R_BAR COMPUTATION
# ============================================================

def mean_grain_size_from_distribution(
    dist_2d,
    bin_centers
):
    """
    Computes the weighted mean grain size:

        R_bar(t) =
            sum_i[p_i(t) * R_i] / sum_i[p_i(t)]

    This formula works with normalized and non-normalized distributions.

    Parameters
    ----------
    dist_2d : ndarray, shape (T, datasize)
        Grain-size distributions.

    bin_centers : ndarray, shape (datasize,)
        Physical bin centers in mm.

    Returns
    -------
    mean_size : ndarray, shape (T,)
        Mean grain size at each time step.
    """

    dist_2d = np.asarray(
        dist_2d,
        dtype=float
    )

    bin_centers = np.asarray(
        bin_centers,
        dtype=float
    )

    denominator = np.sum(
        dist_2d,
        axis=1
    )

    numerator = (
        dist_2d @ bin_centers
    )

    mean_size = np.full_like(
        denominator,
        fill_value=np.nan,
        dtype=float
    )

    valid = denominator != 0.0

    mean_size[valid] = (
        numerator[valid] /
        denominator[valid]
    )

    return mean_size


# ============================================================
# BURKE–TURNBULL TRANSFORMATION
# ============================================================

def burke_turnbull_curve(
    mean_radius,
    initial_radius
):
    """
    Computes:

        R_bar(t)^2 - R_bar(0)^2

    The same R_bar(0) must be used for the ground truth and the models
    to preserve a common physical reference.

    Parameters
    ----------
    mean_radius : ndarray, shape (T,)
        R_bar(t).

    initial_radius : float
        R_bar(0), obtained from the ground truth.

    Returns
    -------
    transformed : ndarray, shape (T,)
    """

    mean_radius = np.asarray(
        mean_radius,
        dtype=float
    )

    transformed = np.full_like(
        mean_radius,
        fill_value=np.nan,
        dtype=float
    )

    valid = np.isfinite(
        mean_radius
    )

    transformed[valid] = (
        mean_radius[valid] ** 2
        - initial_radius ** 2
    )

    return transformed


# ============================================================
# GROUND-TRUTH/PREDICTION METRICS
# ============================================================

def compute_metrics_1d(
    y_true,
    y_pred
):
    """
    Computes:
      - MAE
      - MSE
      - RMSE
      - MRE as a percentage
      - R² between the prediction and the ground truth
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if not np.any(valid):
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MRE%": np.nan,
            "R2": np.nan,
        }

    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]

    difference = (
        y_pred_valid - y_true_valid
    )

    mae = float(
        np.mean(
            np.abs(difference)
        )
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    nonzero_true = (
        y_true_valid != 0.0
    )

    if np.any(nonzero_true):
        mre = float(
            np.mean(
                np.abs(
                    (
                        y_true_valid[nonzero_true]
                        - y_pred_valid[nonzero_true]
                    )
                    / y_true_valid[nonzero_true]
                )
            )
            * 100.0
        )
    else:
        mre = 0.0

    try:
        prediction_r2 = float(
            r2_score(
                y_true_valid,
                y_pred_valid
            )
        )
    except Exception:
        prediction_r2 = float("nan")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MRE%": mre,
        "R2": prediction_r2,
    }


def print_metrics(
    title,
    metrics
):
    """
    Displays the prediction metrics.
    """

    print(
        f"{title}: "
        f"MAE={metrics['MAE']:.6e} | "
        f"MSE={metrics['MSE']:.6e} | "
        f"RMSE={metrics['RMSE']:.6e} | "
        f"MRE={metrics['MRE%']:.2f}% | "
        f"R²_prediction={metrics['R2']:.4f}"
    )


# ============================================================
# BURKE–TURNBULL LINEAR REGRESSION
# ============================================================

def compute_burke_turnbull_fit(
    x,
    y
):
    """
    Fits:

        y = K*x + b

    where:
        y = R_bar(t)^2 - R_bar(0)^2

    For an ideal law passing exactly through the origin,
    the intercept should be close to zero.

    Returns
    -------
    Dictionary containing:
      K
      intercept
      R2_linearity
      n_points
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if np.sum(valid) < 2:
        return {
            "K": np.nan,
            "intercept": np.nan,
            "R2_linearity": np.nan,
            "n_points": int(np.sum(valid)),
        }

    x_valid = x[valid].reshape(
        -1,
        1
    )

    y_valid = y[valid]

    regression = LinearRegression()

    regression.fit(
        x_valid,
        y_valid
    )

    y_fitted = regression.predict(
        x_valid
    )

    try:
        linearity_r2 = float(
            r2_score(
                y_valid,
                y_fitted
            )
        )
    except Exception:
        linearity_r2 = float("nan")

    return {
        "K": float(regression.coef_[0]),
        "intercept": float(regression.intercept_),
        "R2_linearity": linearity_r2,
        "n_points": int(len(y_valid)),
    }


def print_burke_turnbull_fit(
    title,
    fit
):
    """
    Displays the slope K and the linear-fit R².
    """

    print(
        f"{title}: "
        f"K={fit['K']:.6e} | "
        f"intercept={fit['intercept']:.6e} | "
        f"R²_linéarité={fit['R2_linearity']:.4f} | "
        f"N={fit['n_points']}"
    )


# ============================================================
# X-AXIS
# ============================================================

def get_x_axis_full(
    length,
    use_minutes=False,
    dt_min=1.0
):
    """
    Returns either time steps or minutes.
    """

    steps = np.arange(
        length,
        dtype=float
    )

    if use_minutes:
        return steps * dt_min

    return steps


# ============================================================
# PER-SEQUENCE OUTPUT
# ============================================================

def save_sequence_burke_turnbull_plot_and_csv(
    seq_name,
    x_full,
    gt_curve_full,
    pred_curves_full_by_model,
    fit_by_curve,
    out_dir
):
    """
    Saves:
      - the curve CSV file;
      - the curve PNG file;
      - the Burke–Turnbull regression CSV file.
    """

    safe_makedirs(
        out_dir
    )

    # --------------------------------------------------------
    # CURVE CSV FILE
    # --------------------------------------------------------

    curves_csv_path = os.path.join(
        out_dir,
        "burke_turnbull_Rbar2_minus_R0bar2_ALL_MODELS.csv"
    )

    with open(
        curves_csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        header = [
            "x",
            "GT_Rbar2_minus_R0bar2"
        ] + [
            f"{model_type.upper()}_Rbar2_minus_R0bar2"
            for model_type in model_types
            if model_type in pred_curves_full_by_model
        ]

        writer.writerow(
            header
        )

        for index in range(
            len(x_full)
        ):
            row = [
                float(x_full[index]),
                (
                    ""
                    if not np.isfinite(
                        gt_curve_full[index]
                    )
                    else float(
                        gt_curve_full[index]
                    )
                )
            ]

            for model_type in model_types:
                if model_type not in pred_curves_full_by_model:
                    continue

                value = pred_curves_full_by_model[
                    model_type
                ][index]

                row.append(
                    ""
                    if not np.isfinite(value)
                    else float(value)
                )

            writer.writerow(
                row
            )

    # --------------------------------------------------------
    # FIT CSV FILE
    # --------------------------------------------------------

    fits_csv_path = os.path.join(
        out_dir,
        "burke_turnbull_LINEAR_FITS.csv"
    )

    with open(
        fits_csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "curve",
            "K_slope",
            "intercept",
            "R2_linearity",
            "n_points"
        ])

        for curve_name, fit in fit_by_curve.items():
            writer.writerow([
                curve_name,
                fit["K"],
                fit["intercept"],
                fit["R2_linearity"],
                fit["n_points"],
            ])

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        x_full,
        gt_curve_full,
        marker="o",
        label="TRM reference",
        color=colors["gt"]
    )

    for model_type in model_types:
        if model_type not in pred_curves_full_by_model:
            continue

        model_fit = fit_by_curve[
            model_type
        ]

        plt.plot(
            x_full,
            pred_curves_full_by_model[
                model_type
            ],
            marker="o",
            label=labels.get(model_type, model_type.upper()),
            color=colors.get(
                model_type,
                None
            )
        )

    plt.axvline(
        x=x_full[window_size],
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        color="black",
        label="Start of recursive forecasting"
    )

    plt.xlabel(
        "Time (min)"
        if USE_MINUTES
        else "Time step"
    )

    plt.ylabel(
        r"$\overline{R}(t)^2-\overline{R}(0)^2$ (mm$^2$)"
    )

    plt.title(
        ""
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.legend(
        loc="best",
        framealpha=0.9
    )

    plt.tight_layout()

    figure_path = os.path.join(
        out_dir,
        "burke_turnbull_Rbar2_minus_R0bar2_ALL_MODELS.png"
    )

    plt.savefig(
        figure_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SEQ {seq_name}] saved:"
    )

    print(
        f"  - {curves_csv_path}"
    )

    print(
        f"  - {fits_csv_path}"
    )

    print(
        f"  - {figure_path}"
    )


# ============================================================
# GLOBAL OUTPUT
# ============================================================

def save_global_burke_turnbull_plot_and_csv(
    x_full,
    gt_mean_full,
    mean_full_by_model,
    fit_by_curve,
    out_dir
):
    """
    Saves the global mean curves and their regressions.
    """

    safe_makedirs(
        out_dir
    )

    # --------------------------------------------------------
    # MEAN-CURVE CSV FILE
    # --------------------------------------------------------

    curves_csv_path = os.path.join(
        out_dir,
        "burke_turnbull_MEAN_Rbar2_minus_R0bar2_ALL_MODELS.csv"
    )

    with open(
        curves_csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        header = [
            "x",
            "GT_MEAN_Rbar2_minus_R0bar2"
        ] + [
            f"{model_type.upper()}_MEAN_Rbar2_minus_R0bar2"
            for model_type in model_types
            if model_type in mean_full_by_model
        ]

        writer.writerow(
            header
        )

        for index in range(
            len(x_full)
        ):
            row = [
                float(x_full[index]),
                (
                    ""
                    if not np.isfinite(
                        gt_mean_full[index]
                    )
                    else float(
                        gt_mean_full[index]
                    )
                )
            ]

            for model_type in model_types:
                if model_type not in mean_full_by_model:
                    continue

                value = mean_full_by_model[
                    model_type
                ][index]

                row.append(
                    ""
                    if not np.isfinite(value)
                    else float(value)
                )

            writer.writerow(
                row
            )

    # --------------------------------------------------------
    # FIT CSV FILE GLOBAUX
    # --------------------------------------------------------

    fits_csv_path = os.path.join(
        out_dir,
        "burke_turnbull_GLOBAL_LINEAR_FITS.csv"
    )

    with open(
        fits_csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "curve",
            "K_slope",
            "intercept",
            "R2_linearity",
            "n_points"
        ])

        for curve_name, fit in fit_by_curve.items():
            writer.writerow([
                curve_name,
                fit["K"],
                fit["intercept"],
                fit["R2_linearity"],
                fit["n_points"],
            ])

    # --------------------------------------------------------
    # GLOBAL FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        x_full,
        gt_mean_full,
        marker="o",
        label="TRM reference",
        color=colors["gt"]
    )

    for model_type in model_types:
        if model_type not in mean_full_by_model:
            continue

        model_fit = fit_by_curve[
            model_type
        ]

        plt.plot(
            x_full,
            mean_full_by_model[
                model_type
            ],
            marker="o",
            label=model_type.upper(),
            color=colors.get(
                model_type,
                None
            )
        )

    plt.axvline(
        x=x_full[window_size],
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        color="black",
        label="Start of recursive forecasting"
    )

    plt.xlabel(
        "Time (min)"
        if USE_MINUTES
        else "Time step"
    )

    plt.ylabel(
        r"$\overline{R}(t)^2-\overline{R}(0)^2$ (mm$^2$)"
    )

    plt.title(
        ""
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.legend(
        loc="best",
        framealpha=0.9
    )

    plt.tight_layout()

    figure_path = os.path.join(
        out_dir,
        "burke_turnbull_MEAN_Rbar2_minus_R0bar2_ALL_MODELS.png"
    )

    plt.savefig(
        figure_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[GLOBAL MEAN] saved:"
    )

    print(
        f"  - {curves_csv_path}"
    )

    print(
        f"  - {fits_csv_path}"
    )

    print(
        f"  - {figure_path}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "SCRIPT — TEST DE BURKE–TURNBULL"
    )

    print(
        "Grandeur : R_bar(t)^2 - R_bar(0)^2"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # VALIDATION CHECKS
    # --------------------------------------------------------

    if len(BIN_CENTERS) != datasize:
        raise ValueError(
            f"BIN_CENTERS doit avoir {datasize} valeurs, "
            f"reçu {len(BIN_CENTERS)}"
        )

    for model_type in model_types:
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modèle {model_type} non valide. "
                f"Choix possibles : {AVAILABLE_MODELS}"
            )

    if not os.path.exists(
        new_sequences_root
    ):
        raise FileNotFoundError(
            "Dossier new_sequences_root introuvable : "
            f"{new_sequences_root}"
        )

    reset_directory(
        out_root
    )

    # --------------------------------------------------------
    # SEQUENCE LIST
    # --------------------------------------------------------

    sequence_names = []
    sequence_paths = []

    for sequence_name in sorted(
        os.listdir(new_sequences_root),
        key=numerical_sort
    ):
        sequence_path = os.path.join(
            new_sequences_root,
            sequence_name
        )

        if os.path.isdir(
            sequence_path
        ):
            sequence_names.append(
                sequence_name
            )

            sequence_paths.append(
                sequence_path
            )

    if len(sequence_names) == 0:
        raise RuntimeError(
            "Aucune séquence trouvée dans new_sequences_root."
        )

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    models = {}

    for model_type in model_types:

        model_path = os.path.join(
            base_output_dir,
            f"model_{model_type}.keras"
        )

        if not os.path.exists(
            model_path
        ):
            print(
                "[WARN] modèle manquant, ignoré : "
                f"{model_path}"
            )

            continue

        models[model_type] = (
            tf.keras.models.load_model(
                model_path
            )
        )

        print(
            f"✅ Modèle chargé : {model_path}"
        )

    if len(models) == 0:
        raise RuntimeError(
            "Aucun modèle chargé. Vérifie les fichiers "
            "model_<type>.keras dans base_output_dir."
        )

    # --------------------------------------------------------
    # GLOBAL DATA STRUCTURES
    # --------------------------------------------------------

    gt_curves_full_all = []

    pred_curves_full_all_by_model = {
        model_type: []
        for model_type in models.keys()
    }

    global_prediction_metrics_by_model = {
        model_type: []
        for model_type in models.keys()
    }

    global_bt_fits_by_model = {
        model_type: []
        for model_type in models.keys()
    }

    global_gt_bt_fits = []

    print(
        "\n" + "=" * 90
    )

    print(
        "[PAR SÉQUENCE] Courbes Burke–Turnbull + métriques"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # LOOP OVER SEQUENCES
    # --------------------------------------------------------

    for sequence_name, sequence_path in zip(
        sequence_names,
        sequence_paths
    ):

        sequence_normalized, sums, files = (
            load_one_sequence_in_memory(
                sequence_path
            )
        )

        if sequence_normalized is None:
            print(
                f"[SKIP] {sequence_name}: séquence vide."
            )

            continue

        required_length = (
            window_size + output_size
        )

        if len(sequence_normalized) < required_length:
            print(
                f"[SKIP] {sequence_name}: séquence trop courte "
                f"({len(sequence_normalized)}), "
                f"minimum={required_length}"
            )

            continue

        # Use exactly window_size + output_size time steps
        sequence_used = sequence_normalized[
            :required_length
        ]

        # Time axis starting at t=0
        x_full = get_x_axis_full(
            len(sequence_used),
            use_minutes=USE_MINUTES,
            dt_min=DT_MIN
        )

        # ----------------------------------------------------
        # GROUND TRUTH: R_BAR(T)
        # ----------------------------------------------------

        gt_mean_radius = (
            mean_grain_size_from_distribution(
                sequence_used,
                BIN_CENTERS
            )
        )

        if not np.isfinite(
            gt_mean_radius[0]
        ):
            print(
                f"[SKIP] {sequence_name}: R_bar(0) invalide."
            )

            continue

        # Common reference for all curves
        initial_radius = float(
            gt_mean_radius[0]
        )

        # Ground truth: R_bar(t)^2 - R_bar(0)^2
        gt_curve_full = burke_turnbull_curve(
            gt_mean_radius,
            initial_radius
        )

        # Fit the ground truth from t=0
        gt_bt_fit = compute_burke_turnbull_fit(
            x_full,
            gt_curve_full
        )

        global_gt_bt_fits.append(
            gt_bt_fit
        )

        print_burke_turnbull_fit(
            f"[{sequence_name}][GT][BURKE-TURNBULL]",
            gt_bt_fit
        )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        pred_curves_full_by_model = {}
        fit_by_curve = {
            "gt": gt_bt_fit
        }

        for model_type, model in models.items():

            predicted_distributions = (
                autoregressive_predict_norm(
                    model,
                    sequence_normalized,
                    window_size,
                    output_size
                )
            )

            # Predicted R_bar for the output_size time steps
            predicted_mean_radius = (
                mean_grain_size_from_distribution(
                    predicted_distributions,
                    BIN_CENTERS
                )
            )

            # Use the same R_bar(0) reference as the ground truth
            predicted_bt_curve = burke_turnbull_curve(
                predicted_mean_radius,
                initial_radius
            )

            # Complete curve with NaN values before window_size
            predicted_bt_curve_full = np.full(
                len(gt_curve_full),
                np.nan,
                dtype=float
            )

            predicted_bt_curve_full[
                window_size:
                window_size + output_size
            ] = predicted_bt_curve

            pred_curves_full_by_model[
                model_type
            ] = predicted_bt_curve_full

            # ------------------------------------------------
            # PREDICTION/GROUND-TRUTH METRICS
            # ------------------------------------------------

            gt_prediction_region = gt_curve_full[
                window_size:
                window_size + output_size
            ]

            prediction_metrics = compute_metrics_1d(
                gt_prediction_region,
                predicted_bt_curve
            )

            global_prediction_metrics_by_model[
                model_type
            ].append(
                prediction_metrics
            )

            print_metrics(
                f"[{sequence_name}][{model_type.upper()}]",
                prediction_metrics
            )

            # ------------------------------------------------
            # MODEL BURKE–TURNBULL LINEARITY
            # ------------------------------------------------

            predicted_x = x_full[
                window_size:
                window_size + output_size
            ]

            model_bt_fit = compute_burke_turnbull_fit(
                predicted_x,
                predicted_bt_curve
            )

            fit_by_curve[
                model_type
            ] = model_bt_fit

            global_bt_fits_by_model[
                model_type
            ].append(
                model_bt_fit
            )

            print_burke_turnbull_fit(
                (
                    f"[{sequence_name}]"
                    f"[{model_type.upper()}]"
                    f"[BURKE-TURNBULL]"
                ),
                model_bt_fit
            )

        # ----------------------------------------------------
        # PER-SEQUENCE OUTPUT
        # ----------------------------------------------------

        sequence_output_directory = os.path.join(
            out_root,
            "per_sequence",
            sequence_name
        )

        save_sequence_burke_turnbull_plot_and_csv(
            sequence_name,
            x_full,
            gt_curve_full,
            pred_curves_full_by_model,
            fit_by_curve,
            sequence_output_directory
        )

        gt_curves_full_all.append(
            gt_curve_full
        )

        for model_type in models.keys():
            if model_type in pred_curves_full_by_model:
                pred_curves_full_all_by_model[
                    model_type
                ].append(
                    pred_curves_full_by_model[
                        model_type
                    ]
                )

        gc.collect()

    # --------------------------------------------------------
    # POST-PROCESSING CHECK
    # --------------------------------------------------------

    if len(gt_curves_full_all) == 0:
        raise RuntimeError(
            "Aucune séquence valide pour calculer "
            "les courbes Burke–Turnbull."
        )

    # ========================================================
    # GLOBAL MEAN CURVES
    # ========================================================

    gt_curves_full_all = np.stack(
        gt_curves_full_all,
        axis=0
    )

    gt_mean_full = np.nanmean(
        gt_curves_full_all,
        axis=0
    )

    mean_full_by_model = {}

    for model_type, curves_list in (
        pred_curves_full_all_by_model.items()
    ):
        if len(curves_list) == 0:
            continue

        curves_array = np.stack(
            curves_list,
            axis=0
        )

        # NaN values before window_size are intentional
        with np.errstate(
            invalid="ignore"
        ):
            mean_full_by_model[
                model_type
            ] = np.nanmean(
                curves_array,
                axis=0
            )

    x_full_global = get_x_axis_full(
        len(gt_mean_full),
        use_minutes=USE_MINUTES,
        dt_min=DT_MIN
    )

    # Fit the mean curves
    global_fit_by_curve = {
        "gt": compute_burke_turnbull_fit(
            x_full_global,
            gt_mean_full
        )
    }

    print(
        "\n" + "=" * 90
    )

    print(
        "[GLOBAL MEAN] Régressions Burke–Turnbull"
    )

    print(
        "=" * 90
    )

    print_burke_turnbull_fit(
        "[GLOBAL MEAN][GT]",
        global_fit_by_curve["gt"]
    )

    for model_type, mean_curve in (
        mean_full_by_model.items()
    ):
        model_fit = compute_burke_turnbull_fit(
            x_full_global,
            mean_curve
        )

        global_fit_by_curve[
            model_type
        ] = model_fit

        print_burke_turnbull_fit(
            f"[GLOBAL MEAN][{model_type.upper()}]",
            model_fit
        )

    # ========================================================
    # BURKE–TURNBULL SLOPE COMPARISON TABLE
    # ========================================================

    # Fit the TRM reference over the same time interval
    # as the models to compare K consistently.
    gt_fit_prediction_region = compute_burke_turnbull_fit(
        x_full_global[
            window_size:
            window_size + output_size
        ],
        gt_mean_full[
            window_size:
            window_size + output_size
        ]
    )

    gt_K_prediction_region = gt_fit_prediction_region["K"]

    print("\n" + "=" * 100)
    print("BURKE–TURNBULL COMPARISON: LINEARITY AND GROWTH-RATE SLOPES")
    print("=" * 100)

    print(
        f"{'Curve':<18}"
        f"{'K slope':>16}"
        f"{'Linear-fit R²':>18}"
        f"{'K / K_TRM':>16}"
        f"{'Difference (%)':>18}"
    )

    print("-" * 86)

    print(
        f"{'TRM reference':<18}"
        f"{gt_K_prediction_region:>16.6e}"
        f"{gt_fit_prediction_region['R2_linearity']:>18.4f}"
        f"{1.0:>16.4f}"
        f"{0.0:>18.2f}"
    )

    for model_type in model_types:

        if model_type not in global_fit_by_curve:
            continue

        model_fit = global_fit_by_curve[model_type]
        model_K = model_fit["K"]

        if (
            np.isfinite(gt_K_prediction_region)
            and gt_K_prediction_region != 0.0
            and np.isfinite(model_K)
        ):
            slope_ratio = model_K / gt_K_prediction_region
            slope_difference_percent = (
                100.0
                * (model_K - gt_K_prediction_region)
                / gt_K_prediction_region
            )
        else:
            slope_ratio = np.nan
            slope_difference_percent = np.nan

        print(
            f"{model_type.upper():<18}"
            f"{model_K:>16.6e}"
            f"{model_fit['R2_linearity']:>18.4f}"
            f"{slope_ratio:>16.4f}"
            f"{slope_difference_percent:>18.2f}"
        )

    print("=" * 100)

    print(
        "Interpretation: linear-fit R² values close to 1 indicate "
        "nearly linear Burke–Turnbull evolution; differences in K "
        "quantify under- or overestimation of the TRM growth rate."
    )

    global_output_directory = os.path.join(
        out_root,
        "global"
    )

    save_global_burke_turnbull_plot_and_csv(
        x_full_global,
        gt_mean_full,
        mean_full_by_model,
        global_fit_by_curve,
        global_output_directory
    )

    # ========================================================
    # GLOBAL PREDICTION-METRIC SUMMARY
    # ========================================================

    prediction_summary_csv = os.path.join(
        global_output_directory,
        "GLOBAL_PREDICTION_METRICS_BURKE_TURNBULL_BY_MODEL.csv"
    )

    with open(
        prediction_summary_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "model_type",
            "mean_MAE",
            "mean_MSE",
            "mean_RMSE",
            "mean_MRE_percent",
            "mean_R2_prediction"
        ])

        for model_type in model_types:

            if model_type not in (
                global_prediction_metrics_by_model
            ):
                continue

            model_metrics = (
                global_prediction_metrics_by_model[
                    model_type
                ]
            )

            if len(model_metrics) == 0:
                continue

            writer.writerow([
                model_type,
                float(np.nanmean([
                    metric["MAE"]
                    for metric in model_metrics
                ])),
                float(np.nanmean([
                    metric["MSE"]
                    for metric in model_metrics
                ])),
                float(np.nanmean([
                    metric["RMSE"]
                    for metric in model_metrics
                ])),
                float(np.nanmean([
                    metric["MRE%"]
                    for metric in model_metrics
                ])),
                float(np.nanmean([
                    metric["R2"]
                    for metric in model_metrics
                ])),
            ])

    # ========================================================
    # GLOBAL BURKE–TURNBULL LINEARITY SUMMARY
    # ========================================================

    bt_summary_csv = os.path.join(
        global_output_directory,
        "GLOBAL_BURKE_TURNBULL_LINEARITY_SUMMARY.csv"
    )

    with open(
        bt_summary_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "curve",
            "mean_K",
            "std_K",
            "mean_intercept",
            "mean_R2_linearity",
            "std_R2_linearity",
            "number_of_sequences"
        ])

        # Ground truth
        writer.writerow([
            "GT",
            float(np.nanmean([
                fit["K"]
                for fit in global_gt_bt_fits
            ])),
            float(np.nanstd([
                fit["K"]
                for fit in global_gt_bt_fits
            ])),
            float(np.nanmean([
                fit["intercept"]
                for fit in global_gt_bt_fits
            ])),
            float(np.nanmean([
                fit["R2_linearity"]
                for fit in global_gt_bt_fits
            ])),
            float(np.nanstd([
                fit["R2_linearity"]
                for fit in global_gt_bt_fits
            ])),
            len(global_gt_bt_fits)
        ])

        # Models
        for model_type in model_types:

            if model_type not in global_bt_fits_by_model:
                continue

            model_fits = global_bt_fits_by_model[
                model_type
            ]

            if len(model_fits) == 0:
                continue

            writer.writerow([
                model_type,
                float(np.nanmean([
                    fit["K"]
                    for fit in model_fits
                ])),
                float(np.nanstd([
                    fit["K"]
                    for fit in model_fits
                ])),
                float(np.nanmean([
                    fit["intercept"]
                    for fit in model_fits
                ])),
                float(np.nanmean([
                    fit["R2_linearity"]
                    for fit in model_fits
                ])),
                float(np.nanstd([
                    fit["R2_linearity"]
                    for fit in model_fits
                ])),
                len(model_fits)
            ])

    # ========================================================
    # END
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "[GLOBAL] Fichiers de résumé sauvegardés :"
    )

    print(
        f"  - {prediction_summary_csv}"
    )

    print(
        f"  - {bt_summary_csv}"
    )

    print(
        "=" * 90
    )

    tf.keras.backend.clear_session()

    gc.collect()

    print(
        "\n" + "=" * 90
    )

    print(
        "FIN. Sorties générées :"
    )

    print(
        f"  - {out_root}/per_sequence/<seq>/"
    )

    print(
        "      burke_turnbull_Rbar2_minus_R0bar2_ALL_MODELS.png"
    )

    print(
        "      burke_turnbull_Rbar2_minus_R0bar2_ALL_MODELS.csv"
    )

    print(
        "      burke_turnbull_LINEAR_FITS.csv"
    )

    print(
        f"  - {out_root}/global/"
    )

    print(
        "      burke_turnbull_MEAN_Rbar2_minus_R0bar2_ALL_MODELS.png"
    )

    print(
        "      burke_turnbull_MEAN_Rbar2_minus_R0bar2_ALL_MODELS.csv"
    )

    print(
        "      burke_turnbull_GLOBAL_LINEAR_FITS.csv"
    )

    print(
        "      GLOBAL_PREDICTION_METRICS_BURKE_TURNBULL_BY_MODEL.csv"
    )

    print(
        "      GLOBAL_BURKE_TURNBULL_LINEARITY_SUMMARY.csv"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()