import os
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import nibabel as nib
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt

import numpy as np


# Path to the configuration file
CONFIG_FILE = "last_inputs.json"

def load_last_inputs():
    """Load the last used inputs from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as file:
                return json.load(file)
        except Exception:
            return {}
    return {}

def save_last_inputs(inputs):
    """Save the current inputs to a JSON file."""
    try:
        with open(CONFIG_FILE, "w") as file:
            json.dump(inputs, file)
    except Exception as e:
        print(f"Error saving inputs: {e}")



def load_and_resample_nifti(nifti_path):
    """
    Load a NIfTI file and resample it to match the target shape and affine.
    """
    nifti_image = nib.load(nifti_path)   # returns a NIfTI image object that contains both the image data and associated metadata
    nifti_data = nifti_image.get_fdata() # retrieves the image data stored in the NIfTI file and converts it into a NumPy array
    nifti_affine = nifti_image.affine    # Mapping voxel indices to real-world coordinates; Resampling or aligning images; Interpreting image orientation
    # nifti_data(sag,cor,axial,vol)
    return nifti_data


def get_stripe_match_score(twoD_image, periodicity, bandwidth):

    # Perform Fourier Transform
    F = fft2(twoD_image)
    F_shifted = fftshift(F)
    magnitude_spectrum = np.abs(F_shifted)

    # Identify the target frequency index
    freq_index = int(twoD_image.shape[0] / periodicity)

    # Create a mask to isolate the target frequency band
    bandwidth = 5  # Allow a small range around the target frequency
    mask = np.zeros_like(F_shifted, dtype=bool)
    mask[freq_index - bandwidth:freq_index + bandwidth, :] = True

    # Compute the total energy and energy in the target frequency band
    total_energy = np.sum(magnitude_spectrum)
    target_energy = np.sum(magnitude_spectrum[mask])

    # Compute the match score
    match_score = target_energy / total_energy
    
    return match_score

# find match score outliers 
def find_outliers(match_score_input, std_factor):

    #print("scores:", input)
    
    # Convert the list of string values to a list of floats
    scores = [float(score) for score in match_score_input]
    
    #print("scores:", scores)
        
    std_factor = float(std_factor)
    mean_value = np.mean(scores)
    std_value = np.std(scores)
    

    # Identify outliers as those beyond the threshold of mean ± (std_factor * std_value)
    outliers_indices = [
        i for i, score in enumerate(scores) if abs(score - mean_value) > std_factor * std_value
    ]

    return outliers_indices


def plot_match_scores(input, output_path, std_factor, outlier_indices):

    
    #print(f"Mean of match scores: {mean_value}")
    #print("input:", input)
    
    # Convert the list of string values to a list of floats
    scores = [float(score) for score in input]
    #print("scores_for_plot:", scores)

    std_factor = float(std_factor)
    mean_value = np.mean(scores)
    std_value = np.std(scores)
    threshold = std_factor * std_value

    deviation_vector = [score - mean_value for score in scores]


    # Generate x-axis values (assuming sequential order)
    x = list(range(1, len(deviation_vector) + 1))

    # Plot the data
    deviation_vector_abs = np.abs(deviation_vector)
    plt.figure(figsize=(16, 6))  # Double the width (16) compared to the original (8)
    plt.plot(x, deviation_vector_abs, linestyle='-', color='b', label='Match Score')  # Plot match scores

    # Highlight the outliers
    plt.scatter(
        [x[i] for i in outlier_indices],
        [deviation_vector_abs[i] for i in outlier_indices],
        color='red',
        label=f'Outliers (> {std_factor} stds)',
        zorder=5
    )

    
    # Add a horizontal line for the threshold
    plt.axhline(y=threshold, color='r', linestyle='--', label="threshold")
    
    # Title and labels
    # plt.title('Match Scores with Outliers')
    outlier_indices_p1 = [i + 1 for i in outlier_indices] # make outlier index start at one (not zero)
    plt.title(f'Match Scores with Outliers: {outlier_indices_p1}')
    plt.xlabel('Index')
    plt.ylabel('Match Score')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified output path
    plt.savefig(os.path.join(output_path, "match_score_plot.png"))
    print(f"Plot saved to {output_path}")

    plt.draw()  # Draw the plot
    plt.pause(0.001)  # Allow the plot window to update

    # Optionally, show the plot
    #plt.show()


def plot_outlier_images(nifti_data, sagPos, outlier_indices, output_path):
    numOutlier = len(outlier_indices)
    outlier_indices_p1 = [i + 1 for i in outlier_indices]
    
   
    plt.figure(figsize=(10, 5))

    
    for colIdx in range(numOutlier):
        plt.subplot(1, numOutlier, 1 + colIdx)
        plt.imshow(nifti_data[sagPos,:, :, outlier_indices[colIdx]].T, cmap="gray", origin="lower", vmin=0, vmax=1400)
        plt.title("idx: " + str(outlier_indices_p1[colIdx]))
        

    plt.tight_layout()  # Adjust spacing between subplots
    plt.draw()  # Draw the plot
    plt.pause(0.001)  # Allow the plot window to update



    # Save the plot to the specified output path
    plt.savefig(os.path.join(output_path, "oulier_volumes.png"))
    print(f"Plot saved to {output_path}")




# run_analysis() is started by the run button:
###############################################################

def run_analysis():
    # Get user inputs
    input_file = inputNii_file.get()    
    output_folder = output_path.get()  # Get the output folder
    sagittal_slice_idx = slice_idx_entry.get()
    sagittal_slice_idx = int(sagittal_slice_idx)-1  # make plot idx an int and map to python idx (1 becomes 0)
    periodicity = periodicity_entry.get()
    std_factor = std_factor_entry.get()  # Get the std factor input value

    # Validate inputs
    if not input_file or not os.path.isfile(input_file):
        messagebox.showerror("Error", "Invalid input file.")
        return
    if not output_folder or not os.path.isdir(output_folder):  # Validate output folder
        messagebox.showerror("Error", "Invalid output folder.")
        return
    try:
        periodicity = int(periodicity)
        std_factor = float(std_factor)  # Convert std_factor to float
    except ValueError:
        messagebox.showerror("Error", "Slice index, periodicity, and std factor must be valid numbers.")
        return

    # Save the current inputs
    save_last_inputs({
        "inputNii_file": input_file,
        "output_folder": output_folder,  # Save output folder to config
        "sagittal_slice_idx": sagittal_slice_idx,        
        "periodicity": periodicity,
        "std_factor": std_factor,  # Save std factor to config
    })

    nifti_data = load_and_resample_nifti(input_file)
    nifti_data_sagProj = np.sum(nifti_data, axis=0)
    niiShape = nifti_data.shape

    print("Loading and resampling NIfTI dataset...")
    print("Shape of nii array:", niiShape)
    print("Average of the dicom array:", np.mean(nifti_data))    


    nii_proj_match_score = []

    for volIdx in range(niiShape[3]):
        nii_proj_match_score.append(get_stripe_match_score(np.rot90(nifti_data_sagProj[:, :, volIdx]), periodicity, 5))

    outlier_indices = find_outliers(nii_proj_match_score, std_factor)

    plot_match_scores(nii_proj_match_score, output_folder, std_factor, outlier_indices)
    
    plot_outlier_images(nifti_data, sagittal_slice_idx, outlier_indices, output_folder)
    
    

# Load the last inputs
last_inputs = load_last_inputs()

# Create the GUI
root = tk.Tk()
root.title("Find line artefact in nii file")

# Input file
tk.Label(root, text="Input nii-File:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
inputNii_file = tk.StringVar(value=last_inputs.get("inputNii_file", ""))     # use double backslash \\ for defaut path 
tk.Entry(root, textvariable=inputNii_file, width=50).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: inputNii_file.set(filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")]))).grid(row=0, column=2, padx=10, pady=5)

# Output folder
tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
output_path = tk.StringVar(value=last_inputs.get("output_folder", ""))
tk.Entry(root, textvariable=output_path, width=50).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: output_path.set(filedialog.askdirectory())).grid(row=1, column=2, padx=10, pady=5)

# Sagittal slice index
tk.Label(root, text="Sagittal Slice Index plotted:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
slice_idx_entry = tk.Entry(root, width=10)
slice_idx_entry.grid(row=2, column=1, sticky="w", padx=10, pady=5)
slice_idx_entry.insert(0, last_inputs.get("slice_idx_entry", "40"))  # Default value

# Periodicity
tk.Label(root, text="Periodicity:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
periodicity_entry = tk.Entry(root, width=10)
periodicity_entry.grid(row=3, column=1, sticky="w", padx=10, pady=5)
periodicity_entry.insert(0, last_inputs.get("periodicity", "2"))  # Default value

# Std factor
tk.Label(root, text="Std Factor:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
std_factor_entry = tk.Entry(root, width=10)
std_factor_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)
std_factor_entry.insert(0, last_inputs.get("std_factor", "1.2"))  # Default value


# Run button
tk.Button(root, text="Run", command=run_analysis, width=20).grid(row=6, column=1, pady=10)

root.mainloop()
