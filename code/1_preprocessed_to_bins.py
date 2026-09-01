# This preprocessing script converts raw TRM grain-size data into frequency
# distributions. The raw simulation files are not included due to their size.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib


# Use a non-GUI backend to reduce memory usage
matplotlib.use('Agg')

# Define paths
# base_dir = "/media/admin-eyounes/T7/1-Article/trm_base/2x2/Normale/trm"
# save_base_dir =  "/media/admin-eyounes/T7/1-Article/github_code/trm"

# base_dir = "/media/admin-eyounes/T7/1-Article/trm_base/3x3/Normale/trm"
# save_base_dir =  "/media/admin-eyounes/T7/1-Article/github_code/trm"

# base_dir = "/media/admin-eyounes/T7/1-Article/trm_base/4x4/Normale/trm"
# save_base_dir =  "/media/admin-eyounes/T7/1-Article/github_code/trm"

base_dir = "/media/admin-eyounes/T7/1-Article/trm_base/5x5/Normale/trm"
save_base_dir =  "/media/admin-eyounes/T7/1-Article/github_code/trm"


SUFFIX = "" 

# Time parameters
STEP = 6       # 1 minute in the setup
START = 0       
STOP = 366     # For 3 hours, although the model was initially trained for 1 hour and the data were generated for 1 hour
mult = 1
bins=31
os.makedirs(save_base_dir, exist_ok=True)

# First step: compute the global maximum across all files
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

        # Read the "GrainSize" column
        data = pd.read_csv(file_path, delim_whitespace=True, usecols=["GrainSize"], dtype={"GrainSize": "float32"})
        # grain_size = data["GrainSize"].dropna()
        grain_size = data["GrainSize"].dropna() * mult

        if not grain_size.empty:
            max_val = grain_size.max()
            if max_val > global_max:
                global_max = max_val
        del data

# Use int(global_max)+1 to define the upper bound
# global_max_int = global_max
global_max_int = round(global_max, 2)
print ("global_max_int", global_max_int )
global_max_rounded = round(global_max, 2)
print("global_max_rounded", global_max_rounded)

# Define the bins and their centers based on the global maximum
# bins = np.linspace(0, global_max_int, 61)
bins = np.linspace(0, 0.17, 31)
# bins = np.linspace(0, 0.14, bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Second step: generate histograms using the defined bins
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

        # Read the "GrainSize" column
        data = pd.read_csv(file_path, delim_whitespace=True, usecols=["GrainSize"], dtype={"GrainSize": "float32"})
        # grain_size = data["GrainSize"].dropna()
        grain_size = data["GrainSize"].dropna() * mult

        del data

        if grain_size.empty:
            continue

        # Compute the histogram and weighted mean
        freq, _ = np.histogram(grain_size, bins=bins)
        grain_size_mean  = grain_size.mean()
        # print("Exact mean for this time step:", grain_size_mean_exact)
        # grain_size_mean = np.sum(bin_centers * freq) / np.sum(freq) if np.sum(freq) > 0 else 0

        # Create the histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(grain_size, bins=bins, kde=False, color='blue', edgecolor='black', alpha=0.7)
        plt.plot(bin_centers, freq, marker='o', color='green', linestyle='-', label='Courbe lissée des fréquences')
        plt.axvline(grain_size_mean, color='red', linestyle='--', label=f'Taille moyenne = {grain_size_mean:.4f} mm')
        plt.xlabel('GrainSize')
        plt.ylabel('Fréquence')
        plt.title(f'Histogramme des valeurs de GrainSize - {file_name}')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, 0.17)
        plt.ylim(0, 500)

        # Save the histogram image
        histogram_path = os.path.join(save_dir, f'{file_name.replace(".txt", "")}_histogram.png')
        plt.savefig(histogram_path)
        plt.close()

        # Save the frequency data to a text file
        frequency_file_path = os.path.join(save_dir, f'{file_name.replace(".txt", "")}_frequency.txt')
        with open(frequency_file_path, 'w') as f:
            f.write("Frequency\n")
            f.write("\n".join(map(str, freq)))

        del grain_size, freq


# Use int(global_max)+1 to define the upper bound
# global_max_int = global_max
global_max_int = round(global_max, 2)
print ("global_max_int", global_max_int )
global_max_rounded = round(global_max, 2)
print("global_max_rounded", global_max_rounded)
















# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# import matplotlib

# # Backend without a graphical user interface
# matplotlib.use("Agg")

# # ===============================
# # PATHS
# # ===============================
# base_dir = "/media/admin-eyounes/T7/Data/test2/Case2x2_N_S2_M18/Results/TimeEvolution"
# increments_dir = os.path.join(base_dir, "Increments")

# save_base_dir = "/home/admin-eyounes/Desktop/Pour_article"
# save_dir = os.path.join(save_base_dir, "Case2x2_N_S2_M18")
# os.makedirs(save_dir, exist_ok=True)

# # ===============================
# # TIME PARAMETERS
# # ===============================
# STEP = 1
# START = 0
# STOP = 180   # Adjust if necessary

# # ===============================
# # 1) FIRST PASS: GLOBAL MAXIMUM
# # ===============================
# global_max = 0.0

# files_to_plot = [f"SurfaceData_0_{i}.txt" for i in range(START, STOP, STEP)]

# for file_name in files_to_plot:
#     file_path = os.path.join(increments_dir, file_name)
#     if not os.path.isfile(file_path):
#         continue

#     data = pd.read_csv(
#         file_path,
#         delim_whitespace=True,
#         usecols=["GrainSize"]
#     )

#     grain_size = data["GrainSize"].dropna()

#     if not grain_size.empty:
#         max_val = grain_size.max()
#         if max_val > global_max:
#             global_max = max_val

#     del data, grain_size

# print("global_max =", global_max)

# if global_max == 0:
#     raise RuntimeError("ERREUR : global_max = 0 → aucun GrainSize lu")

# # ===============================
# # COMMON BINS
# # ===============================
# bins = np.linspace(0, global_max, 31)  # 30 bins
# bin_centers = (bins[:-1] + bins[1:]) / 2

# # ===============================
# # 2) HISTOGRAMS
# # ===============================
# for file_name in files_to_plot:
#     file_path = os.path.join(increments_dir, file_name)
#     if not os.path.isfile(file_path):
#         continue

#     data = pd.read_csv(
#         file_path,
#         delim_whitespace=True,
#         usecols=["GrainSize"]
#     )

#     grain_size = data["GrainSize"].dropna()
#     del data

#     if grain_size.empty:
#         continue

#     # Numerical histogram
#     freq, _ = np.histogram(grain_size, bins=bins)

#     # Mean
#     grain_size_mean = grain_size.mean()

#     # Figure
#     plt.figure(figsize=(8, 5))
#     sns.histplot(
#         grain_size,
#         bins=bins,
#         kde=False,
#         color="blue",
#         edgecolor="black",
#         alpha=0.7
#     )

#     plt.plot(
#         bin_centers,
#         freq,
#         marker="o",
#         color="green",
#         linestyle="-",
#         label="Fréquence par bin"
#     )

#     plt.axvline(
#         grain_size_mean,
#         color="red",
#         linestyle="--",
#         label=f"Moyenne = {grain_size_mean:.4e} (unités simulation)"
#     )

#     plt.xlabel("GrainSize (unités simulation)")
#     plt.ylabel("Fréquence")
#     plt.title(f"Histogramme GrainSize – {file_name}")
#     plt.grid(True)
#     plt.legend()

#     plt.xlim(0, global_max)
#     plt.ylim(0, max(freq) * 1.1)

#     # Save image
#     histogram_path = os.path.join(
#         save_dir,
#         file_name.replace(".txt", "_histogram.png")
#     )
#     plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     # Save frequencies
#     freq_path = os.path.join(
#         save_dir,
#         file_name.replace(".txt", "_frequency.txt")
#     )

#     with open(freq_path, "w") as f:
#         f.write("Frequency\n")
#         for v in freq:
#             f.write(f"{v}\n")

#     del grain_size, freq

# print("Traitement terminé avec succès.")
# print ("global_max", global_max)