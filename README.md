This script identifies a movement artefact in nii-BOLD images; a periodic stripe artefact is found on the basis of an FFT and the expected frequency of the stripes, coming from the slice timing 




# install on Windows:

- download python: https://www.python.org/downloads/
- install python (see screenshot install_python_windows.png)
- check Installation via the command line: python --Version
- install necessary libraries via pip: pip install nibabel scipy matplotlib
- download the code to a certain folder
- go to that folder in the windows command line: cd c:\path\folder\where\the\script\is
- once you are there type: python main_gui_FFT_nii_image_analysis.py
  -> the GUI starts (see screenshot GUI.png)




input: 
 - Input nii-File
 - Output Folder: where the Output plots are saved
 - Sagittal Slice Index plotted: the index that will be shown in the outlier plot
 - Periodicity: this is the periodicity of the slices, which can be obtained from the dcm2niix-slice-times;
   Here is an example Output of dcm2niix (https://github.com/rordenlab/dcm2niix) for the SliceTiming:
	"SliceTiming": [
	0,
	0.177,
	0.355,
	0.533,
	0.71,
	0.06,
	0.238,
	...]
   The slice at time '0' ms is obtained first; the slice at time '0.06' ms is obtained second; the distance between them is 5, so that is the periodicity
 - Std Factor: the number of stds from the mean to identify outlier volumes with stripe pattern; this determines the threshold in the match score plot, which determined whas is considered an outlier

output: 
 - match_score_plot.png: a plot of the scores for each volume; the outliers are marked and mentioned in the plot Headline
 - oulier_volumes.png: each volume is plotted that is considered an outlier

