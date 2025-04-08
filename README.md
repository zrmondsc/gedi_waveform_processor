# 🌲 gedi_waveform_processor

A lightweight Python library for filtering, processing, and exporting GEDI L1B waveforms for deep learning workflows.

Designed to work with GEDI L1B data that have been processed from HDF5 to .geojson with NASA's GEDI_Subsetter tool. 

HDF5 functionality has been removed

## 📦 What's in the box?

### 🗂️ `gedi_waveform_processor/`  
Python library to :
- Filter GEDI L1B shots using quality flags
- Decode, normalize, and resample waveform returns
- Visualize and export processed waveforms for ML
- Export `.npz`, `.pt`, or `.h5` datasets with waveform and metadata

---

### 🗂️ `notebooks/`  
Short demos and reproducible experiments:
- `demo.ipynb`: Intro to GEDI waveform processing with the library
- `convolutional_autoencoder_demo.ipynb`: Baseline CAE training + reconstruction workflow
- `convolutional_autoencoder_experiments.ipynb`: Automated experiments for Latent space tuning, dropout, batchnorm, bottleneck type, logging, and visual outputs

---

### 🗂️ `models/`  
Saved autoencoder models and embeddings:
- `autoencoder_exp_xx.keras`: Sample trained model
- `embeddings_exp_xx.npy`: Latent vectors
- `encoder_exp_xx.keras` : Trained encoder from the model
- `experiment_results.csv` : .csv file synthesizing the logged experiment results
- `latent_shape_experiment_results.csv`: .csv file synthesizing logged results of latent shape experiments
---

### 🗂️ `data/`  
Sample waveform data for quick testing:
- `gedi_waveforms_tf.npz`: Processed waveform + metadata
- `sample_data.geojson`: Raw GEDI shots in GeoJSON format (subset from GEDI_Subsetter, merged)
- `encoded_latents.npz`: encoded GEDI waveforms (8 dims)
- `encoded_latents_4x4,npz`: encoded GEDI waveforms (4,4)

---

### 🗂️ `plots/`  
.png files of waveform reconstruction comparisons:
- `selected_indices.npy`: randomly selected waveform indices to plot
- `exp_01_reconstructions_4x4.png`: reconstruction plots for latent shape experiment 01
- `exp_01_reconstructions_4x4.png`: reconstruction plots for latent shape experiment 01
- `exp_01_reconstructions.png`: reconstruction plots for experiment 01
- `exp_02_reconstructions.png`: reconstruction plots for experiment 02


## Installation
```bash
# Create and activate a new conda env
conda env create -f environment.yml
conda activate gedi_pro_env
```

```bash
# install package
pip install git+https://github.com/zrmondsc/gedi_waveform_processor.git
```

