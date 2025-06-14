# HuberEtAl2025_1_RapidFFR
# EEG Analysis Pipeline

A comprehensive multi-language pipeline for processing and analysing EEG data, implementing preprocessing, spectral analysis, and statistical visualisation across MATLAB, Python, and R.

## Overview

This pipeline processes EEG data from BDF files through a complete analysis workflow, including preprocessing, trial segmentation, spectral analysis with bootstrapping, and statistical visualisation. The pipeline is designed for Frequency-Following Response (FFR) analysis and implements robust noise estimation procedures.

## Pipeline Architecture

### 1. Data Preprocessing (MATLAB)
**File:** `unified_ffr_processor.m`

- **Input:** Raw EEG data in BDF format
- **Processing:**
  - Uses EEGLAB for data extraction from BDF files
  - Applies bandpass filtering between 70Hz and 2000Hz
- **Output:** Preprocessed data exported as `.mat` files

### 2. Trial Segmentation (MATLAB)
**File:** `unified_ffr_analyser.m`

- **Input:** Preprocessed CSV files from step 1
- **Processing:**
  - Segments EEG data into individual trials
  - Excludes trials with amplitude exceeding 35µV
  - Applies baseline correction for conventional FFR analysis
- **Output:** Segmented trial data as CSV files

### 3. Spectral Analysis (Python)
**Files:** `ffr_spectral_analyser_exp1.py`, `ffr_spectral_analyser_exp2.py`

- **Input:** Segmented trial CSV files from step 2
- **Processing:**
  - Performs spectral analysis on ERP data
  - Implements bootstrapping procedure for noise estimation
  - Executes 10,000 permutations (optimised for SNR stability at second decimal place)
- **Output:** Spectral analysis results and SNR estimates

### 4. Statistical Analysis & Visualisation (R)

- **Input:** Spectral analysis results from Python from step 3
- **Processing:** Statistical modelling and hypothesis testing
- **Output:** Statistical results and publication-ready figures

## Data Access

### Raw Data
Raw BDF files (>100GB total) are available upon request due to file size constraints. Please contact the author directly for access.

### Processed Data
Preprocessed `.mat` files (output of step 1) are available on OSF: [doi:10.17605/OSF.IO/Q7STZ](https://doi.org/10.17605/OSF.IO/Q7STZ)

## System Requirements

### Software Dependencies
- **MATLAB:** EEGLAB toolbox required
- **Python:** NumPy, Pandas
- **R:** tidyverse, ggpubr, irr, broom, nlme, afex, lme4

### Hardware Recommendations
**For Python Scripts Only**

For bootstrap computations in the Python scripts, parallel processing is recommended. On personal machines, we advise using a maximum of 3 CPU cores to prevent overheating. For cluster computing environments, the number of CPUs can be scaled according to the cluster's specifications.

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd ffr_analysis_pipeline
   ```

2. **Install MATLAB dependencies**
   - Install EEGLAB: https://sccn.ucsd.edu/eeglab/
   - Add EEGLAB to MATLAB path

3. **Install Python dependencies**
   ```bash
   pip install -r python_requirements.txt
   ```

4. **Install R dependencies**
   ```r
   install.packages(c("tidyverse", "ggpubr", "irr", "broom", "nlme", "afex", "lme4"))
   ```

## File Structure

```
ffr-analysis-pipeline/
├── matlab/
│   ├── unified_ffr_processor.m
│   └── unified_ffr_analyser.m
├── python/
│   ├── ffr_spectral_analyser_exp1.py
│   └── ffr_spectral_analyser_exp2.py
├── r/
│   ├── exp1/
│   │   ├── scripts/
│   │   │   ├── DataPrepExp1.Rmd
│   │   │   ├── PlotsExp1.Rmd
│   │   │   ├── H1_RapidVsConv.Rmd
│   │   │   ├── H2_ICCRapidConv.Rmd
│   │   │   └── H3_TestRetest.Rmd
│   │   ├── data/
│   │   │   ├── Signal_Outliers.csv
│   │   │   ├── SignalIterationsMatrix.csv
│   │   │   ├── SignalIterationsMatrixClean.csv
│   │   │   ├── SNR_Outliers.csv
│   │   │   ├── SNRIterationsMatrix.csv
│   │   │   └── SNRIterationsMatrixClean.csv
│   │   └── plots/
│   └── exp2/
│       ├── scripts/
│       │   ├── DataPrepExp2.Rmd
│       │   ├── PlotsExp2.Rmd
│       │   └── H4_LongRapidChunks.Rmd
│       ├── data/
│       │   ├── Signal_Outliers.csv
│       │   ├── SignalIterationsMatrix.csv
│       │   ├── SignalIterationsMatrixClean.csv
│       │   ├── SNR_Outliers.csv
│       │   ├── SNRIterationsMatrix.csv
│       │   └── SNRIterationsMatrixClean.csv
│       └── plots/
├── data/
│   ├── raw/             # BDF files (not included)
│   ├── preprocessed/    # MAT files from preprocessing (not included)
│   ├── processed/       # CSV files from processing (not included)
│   └── results/         # CSV files for R analysis (not included)
├── python_requirements.txt
└── README.md
```

## Citation

If you use this pipeline in your research, please cite:

**TBC**

## Contact

For questions, issues, or raw data access, please contact:

**Jonas Huber**  
Chandler House, 2 Wakefield Street  
London WC1N 1PF, United Kingdom  
Email: jonas.huber.20@ucl.ac.uk
