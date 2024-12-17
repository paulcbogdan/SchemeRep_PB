## Intrinsic fluctuations in global connectivity reflect transitions between states of high and low prediction error

The code for the present report is organized into six folders: `Study1A`, `Study1B`, `Study2A`, `Study2B`, `Study3`, and `Utils` (utility functions). All of the code was written in Python, except for one .R function used for the multilevel regression in `Study2B`. 

Scripts ending in "funcs" contain functions called by other scripts and are not meant to be run. Scripts containing the term "analyze" were used to generate numbers that were reported in the manuscript's text (e.g., the t-value in Study 2B). Scripts containing the term ("plot") were used to generate the figures, although these itself generally contain considerable code for analysis (e.g., computing regressions or calling modularity functions).



### `Study1A` code:
* `load_Study1A_funcs.py`: loads the Study1A task-fMRI data. Note that the data itself cannot be released due to IRB restrictions. Nonetheless, this script has been included. It loads .nii images, extract ROI timeseries, and compute connectivity. Some scripts for the organization of trial data (i.e., defining which trial is high PE and which is low PE) were not included.
* `modularity_funcs.py`: contains functions related to computing modularity and partitioning connectome matrices.
* `plot_Fig2AB_matrices.py`: generates the Figure 2A & 2B matrices via linear regression. Note that many 
* `plot_Fig2CD_partitions.py`: generates the Figure 2C & 2D modules and partitions of the ROIs, in part, with functions from modularity_funcs.py
* `plot_Fig2E_partitions.py`: generates the Figure 2E anatomical modules, in part, with functions from `plot_Fig2CD_partitions.py`.
* `plot_Fig2F_bars.py`: generates the bar graphs, using functions from several other `Study1A` scripts.

### `Study1B` files:
* `analyzie_plot_Fig3.py`: runs all of the Study 1B statistical tests and generates Figure 3. Uses data that was preprocessed via `preprocess_Study1B.py`
* `final_HCP_subjects.txt`: is a list of IDs corresponding to the 1,000 Human Connectome Project (HCP) subjects used. In general, these list corresopnds to the lowest 1000 IDs from the dataset, but participants who did not have data for either run of the task or rs-fMRI session were not included (this could be partly due to errors in downloading).
* `preprocess_Study1B.py`: preprocesses (unzipped) files for the gambling task downloaded from the HCP and runs the single-trial-beta regressions. Preprocessing and single-trial beta code has been released given that this dataset is publicly available. Preprocessing is done using NiLearn.

### `Study2A` files:
* `analyze_plot_Fig4B.py`: runs all of the Study 2A statistical tests and generates the matrix for Figure 4B. Note that the cuboids in Figure 4 and many of the diagram's features were prepared manually.
* `load_Study2A_funcs.py`: loads the rs-fMRI data, which would be used for `analyze_plot_Fig4B.py`.
* `rs_connectivity_funcs.py`: contains several functions related to edge-edge correlations and for organization of the data, which would be used for `analyze_plot_Fig4B.py`.

### `Study2B` files:
* `analyze_multilevel_regression.R`: contains the lme4 code used for the multilevel regression. 
* `analyze_Study2B.py`: computes the task-fMRI and rs-fMRI values from each participant and ROI set. This also runs the t-test on the correlation coefficients. These task-fMRI and rs-fMRI value were organized into a .csv (based on the saved .pkl), which would be used in `analyze_multilevel_regression.py`.
* `plot_Fig5B_histogram.py`: plots the histogram shown in Figure 5B.
* `preprocess_Study2B.py`: preprocesses (unzipped) resting-state files from the HCP. The preprocessed data was used in `analyze_Study2B.py`.

### `Study3` files:
* `analyze_plot_Fig7.py`: runs the Study 3 analysis, generates the values for Table 1 and produces Figure 7.
* `EEG_simultaneous_funcs.py`: contains the functions for analyzing the EEG aspects of the Study 3 dataset.
* `fMRI_simultaneous_funcs.py`: contains the functions for analyzing the fMRI aspects of the Study 3 dataset.
* `plot_Fig7_funcs.py`: contains the functions for generating the Figure 7 plots.
* `simulate_fMRI_EEG.py`: contains the code used for the Study 3 Supplemental Materials simulations.

### `Utils` files:
* `atlas_funcs.py`: functions related to using the Brainnetome Atlas.
* `pickle_wrap_funcs.py`: functions that help with saving and caching data.
* `plotting_funcs.py`: functions related to plotting the functional connectivity matrices.
