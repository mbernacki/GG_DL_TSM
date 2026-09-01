# Grain-Growth Forecasting Using Deep Learning Time-Series Models

This repository contains the processed data, trained models, and Python scripts used to forecast grain-size distribution evolution with RNN, LSTM, TCN, and Transformer models.

The raw data were generated using the high-fidelity **ToRealMotion (TRM)** code. They are not included in this repository because of their large size. The processed 30-bin distributions used by the models are provided in `trm/`.

## Project Structure

```text
code/
├─ 1_preprocessed_to_bins.py
│    ├─ Reads the raw files containing GrainSize values
│    ├─ Converts each raw file into a distribution composed of 30 bins
│    └─ Saves the processed .txt files used by the forecasting models
│
├─ SN_main.py
│    ├─ Loads and normalizes the processed sequences
│    ├─ Generates the training, validation, and test datasets
│    ├─ Trains four Deep Learning models:
│    │      → RNN
│    │      → LSTM
│    │      → TCN
│    │      → Transformer
│    ├─ Performs autoregressive predictions over multiple time steps
│    ├─ Evaluates the predictions using MAE, MSE, RMSE, MRE, and R²
│    └─ Saves the trained models
│
├─ SN_main2.py
│    ├─ Performs 10-fold cross-validation at the sequence level
│    ├─ Trains the selected model for each fold
│    └─ Saves the fold models and performance summaries
│
├─ SN_GSD.py
│    └─ Compares the predicted and reference grain-size distributions
│
├─ SN_grain_count.py
│    └─ Evaluates the sum of the normalized frequencies over time
│
├─ SN_MeanGrainSize.py
│    └─ Computes and compares the mean grain size over time
│
├─ SN_MRE.py
│    └─ Computes the mean relative error at each forecasting time step
│
├─ SN_R2.py
│    └─ Computes R² at each forecasting time step
│
└─ SN_quandratique.py
     └─ Evaluates the Burke–Turnbull grain-growth law and its linearity

models_output/
├─ 1heure/
│    └─ RNN, LSTM, TCN, and Transformer models for the 1-hour forecast
│
└─ 3heures/
     └─ RNN, LSTM, TCN, and Transformer models for the 3-hour forecast

test_sequences/
└─ Six processed sequences used for model evaluation

trm/
└─ Processed 30-bin distributions used for model training

README.md
└─ Project description and usage instructions
```

## Requirements

```bash
python3 -m pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## Usage

Before running a script, update its input and output paths according to the location of the repository on your computer.

### Data preprocessing

This step requires the raw TRM data, which are not included in the repository:

```bash
python3 code/1_preprocessed_to_bins.py
```

The processed files already provided in `trm/` can be used directly without repeating this step.

### Model training and prediction

```bash
python3 code/SN_main.py
```

### Ten-fold cross-validation

```bash
python3 code/SN_main2.py
```

### Additional analyses

```bash
python3 code/SN_GSD.py
python3 code/SN_grain_count.py
python3 code/SN_MeanGrainSize.py
python3 code/SN_MRE.py
python3 code/SN_R2.py
python3 code/SN_quandratique.py
```
