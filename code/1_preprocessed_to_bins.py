import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib


# Utilisation du backend non-GUI pour économiser de la mémoire
matplotlib.use('Agg')

# Définir les chemins
base_dir = "/home/admin-eyounes/Desktop/trm"
save_base_dir = "/home/admin-eyounes/Desktop/trm2"
SUFFIX = "" 

# >>> Paramètres temporels
STEP = 6       # 1 minute dans le setup
START = 0       
STOP = 1080     # (pour 3 heures mais le modele tous d'abors etais entrainé pour une heure et les données etais generer pour une heure))

os.makedirs(save_base_dir, exist_ok=True)

# Première étape : calculer le maximum global parmi tous les fichiers
global_max = 0
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir, "Increments")
    if not os.path.isdir(subdir_path):
        continue

    # files_to_plot = [f"SurfaceData_0_{i}.txt" for i in range(0, 366, 6)]
    files_to_plot = [f"SurfaceData_0_{i}.txt" for i in range(START, STOP, STEP)]
    for file_name in files_to_plot:
        file_path = os.path.join(subdir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        # Lecture de la colonne "GrainSize"
        data = pd.read_csv(file_path, delim_whitespace=True, usecols=["GrainSize"], dtype={"GrainSize": "float32"})
        grain_size = data["GrainSize"].dropna()
        if not grain_size.empty:
            max_val = grain_size.max()
            if max_val > global_max:
                global_max = max_val
        del data

# On utilise int(global_max)+1 pour définir la borne supérieure
# global_max_int = global_max
global_max_int = round(global_max, 2)
print ("global_max_int", global_max_int )
global_max_rounded = round(global_max, 2)
print("global_max_rounded", global_max_rounded)

# Définir les bins et leurs centres en fonction du maximum global
# bins = np.linspace(0, global_max_int, 61)
bins = np.linspace(0, global_max, 31)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Deuxième étape : génération des histogrammes en utilisant les bins définis
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir, "Increments")
    if not os.path.isdir(subdir_path):
        continue
    save_dir = os.path.join(save_base_dir, f"{subdir}{SUFFIX}")
    # save_dir = os.path.join(save_base_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    files_to_plot = [f"SurfaceData_0_{i}.txt" for i in range(START, STOP, STEP)]

    for file_name in files_to_plot:
        file_path = os.path.join(subdir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        # Lecture de la colonne "GrainSize"
        data = pd.read_csv(file_path, delim_whitespace=True, usecols=["GrainSize"], dtype={"GrainSize": "float32"})
        grain_size = data["GrainSize"].dropna()
        del data

        if grain_size.empty:
            continue

        # Calcul de l'histogramme et de la moyenne pondérée
        freq, _ = np.histogram(grain_size, bins=bins)
        grain_size_mean  = grain_size.mean()
        # print("Moyenne exacte pour ce pas de temps :", grain_size_mean_exact)
        # grain_size_mean = np.sum(bin_centers * freq) / np.sum(freq) if np.sum(freq) > 0 else 0

        # Création de l'histogramme
        plt.figure(figsize=(8, 5))
        sns.histplot(grain_size, bins=bins, kde=False, color='blue', edgecolor='black', alpha=0.7)
        plt.plot(bin_centers, freq, marker='o', color='green', linestyle='-', label='Courbe lissée des fréquences')
        plt.axvline(grain_size_mean, color='red', linestyle='--', label=f'Taille moyenne = {grain_size_mean:.4f} mm')
        plt.xlabel('GrainSize')
        plt.ylabel('Fréquence')
        plt.title(f'Histogramme des valeurs de GrainSize - {file_name}')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, global_max_int)
        plt.ylim(0, 500)

        # Sauvegarde de l'image de l'histogramme
        histogram_path = os.path.join(save_dir, f'{file_name.replace(".txt", "")}_histogram.png')
        plt.savefig(histogram_path)
        plt.close()

        # Sauvegarde des données de fréquence dans un fichier texte
        frequency_file_path = os.path.join(save_dir, f'{file_name.replace(".txt", "")}_frequency.txt')
        with open(frequency_file_path, 'w') as f:
            f.write("Frequency\n")
            f.write("\n".join(map(str, freq)))

        del grain_size, freq


# On utilise int(global_max)+1 pour définir la borne supérieure
# global_max_int = global_max
global_max_int = round(global_max, 2)
print ("global_max_int", global_max_int )
global_max_rounded = round(global_max, 2)
print("global_max_rounded", global_max_rounded)
