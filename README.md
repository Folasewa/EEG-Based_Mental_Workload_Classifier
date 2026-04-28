# 🧠 Is Your Brain Overloaded? — EEG-Based Mental Workload Classifier

A machine learning pipeline that detects **binary mental workload states (low vs. high)** from EEG signals, achieving **82.7% accuracy** with a Support Vector Machine using only 15 carefully selected spectral features.

---

## 📌 Overview

Mental workload (MWL) is a measure of how much cognitive effort a task demands. Identifying when someone is cognitively overloaded — in real time — has applications in aviation, healthcare, education, and human-computer interaction.

This project builds an end-to-end EEG classification pipeline:
- Raw EEG → Preprocessing → Feature Extraction → Feature Selection → Classification

It uses **Power Spectral Density (PSD)** features across four EEG frequency bands and a **cross-method feature selection strategy** (NCA + Random Forest) to identify the 15 most discriminative features before classification.

---

## 📊 Results

| Feature Set | Model | CV Accuracy | Cohen's Kappa |
|---|---|---|---|
| **15 Consensus Features** | **SVM (RBF)** | **82.7%** | **0.65** |
| 15 Consensus Features | Random Forest | 79.5% | 0.59 |
| 15 Consensus Features | XGBoost | 79.6% | 0.59 |
| 24 NCA Features | SVM (RBF) | 80.4% | 0.61 |
| 24 NCA Features | Random Forest | 78.3% | 0.57 |
| 24 NCA Features | XGBoost | 77.3% | 0.55 |

> Evaluated under **stratified 5-fold cross-validation** on 93 recordings from 47 participants.

---

## 🗂️ Project Structure

```
EEG-Based_Mental_Workload_Classifier/
│
├── data/                        # Raw EEG recordings (.txt), one per subject per condition
│   ├── sub01_hi.txt
│   ├── sub01_lo.txt
│   └── ...
│
├── preprocessed_data/           # Cleaned EEG arrays saved as .npy files
│   ├── sub01_hi.npy
│   ├── sub01_lo.npy
│   └── ...
│
├── features.csv                 # Extracted PSD features (93 × 58 including subject & workload)
├── ratings.txt                  # Participant subjective workload ratings (1–9 scale)
│
├── exploratory.ipynb            # Data exploration and signal visualisation
├── preprocessing.ipynb          # Full preprocessing pipeline (filter → ASR → re-reference)
├── feature_extraction.ipynb     # PSD feature extraction with sliding windows
└── classification.ipynb         # Feature selection, regression, and binary classification
```

---

## 🔬 Pipeline

### 1. Exploratory Analysis (`exploratory.ipynb`)
- Loads all 96 raw `.txt` recordings (14 channels × 19,200 samples at 128 Hz)
- Visualises raw EEG signals across all 14 channels
- Compares low vs. high workload signals side-by-side for all channels
- Confirms uniform recording duration (150 seconds per recording)

### 2. Preprocessing (`preprocessing.ipynb`)
Follows the Makoto preprocessing pipeline using **MNE-Python** and **ASRpy**:

| Step | Method | Details |
|---|---|---|
| High-pass filter | Butterworth (MNE) | 1 Hz cutoff — removes DC drift |
| Notch filter | MNE notch_filter | 60 Hz — removes line noise |
| Artifact removal | ASR (cutoff = 5) | Removes eye blinks, muscle artifacts |
| Re-referencing | Average reference | Removes global noise |

3 of 96 recordings failed during ASR and were excluded → **93 recordings retained**.

### 3. Feature Extraction (`feature_extraction.ipynb`)
- **Sliding window**: 512-sample windows (4 s), 128-sample step (1 s)
- **PSD**: Welch's method (`nperseg=256`) per window per channel
- **Bands**: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz)
- **Averaging**: PSD features are averaged across all windows per recording
- **Output**: 14 channels × 4 bands = **56 features per recording** → saved to `features.csv`

### 4. Classification (`classification.ipynb`)

**Feature Selection — Consensus of NCA + Random Forest:**
- Both methods run over 5-fold CV on the training set
- Features are ranked by cumulative weight (75% threshold)
- NCA selects 24 features; Random Forest selects its own subset
- The **15-feature intersection** is used for final classification

**Most discriminative features:** Delta and theta band power at frontal (AF3, F7, F3) and parieto-occipital (P7, O1, P8) sites.

**Classifiers evaluated:** SVM (RBF kernel), Random Forest, XGBoost  
**Best model:** SVM with C=100, RBF kernel, trained on 15 consensus features

---

## 🛠️ Setup & Installation

**Python 3.8+** required.

```bash
# Clone the repository
git clone https://github.com/your-username/EEG-Based_Mental_Workload_Classifier.git
cd EEG-Based_Mental_Workload_Classifier

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost mne asrpy
```

---

## ▶️ Usage

Run the notebooks **in order**:

```bash
# 1. Explore the raw data
jupyter notebook exploratory.ipynb

# 2. Preprocess all recordings
jupyter notebook preprocessing.ipynb

# 3. Extract PSD features
jupyter notebook feature_extraction.ipynb

# 4. Run feature selection and classification
jupyter notebook classification.ipynb
```

> **Note:** Update the file paths in each notebook to point to your local data directory before running.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `scipy` | Welch PSD, signal filtering |
| `mne` | EEG preprocessing (filtering, re-referencing) |
| `asrpy` | Artifact Subspace Reconstruction |
| `scikit-learn` | NCA, StandardScaler, SVM, Random Forest, cross-validation |
| `xgboost` | XGBoost classifier |
| `matplotlib`, `seaborn` | Visualisation |

---

## 📁 Dataset

This project uses an open-access EEG dataset from **50 male graduate students at Nanyang Technological University**.

- **Task:** SIMKAP multitasking test (18-minute dual-task paradigm)
- **Headset:** Emotiv EPOC (14 channels, 128 Hz)
- **Electrodes:** AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **Labels:** Participant-rated subjective workload (1–9 scale), binarised at threshold 4

---

## 📄 Paper

This project accompanies the paper:

> **"Is Your Brain Overloaded? Detecting Mental Workload from EEG Signals"**  
> Folasewa Abdulsalam

Key findings:
- Binary workload classification is tractable from consumer-grade EEG with PSD features alone
- A compact 15-feature set (delta/theta, frontal/occipital) outperforms a larger 24-feature set
- Three-class classification (low/moderate/high) degrades to near-chance — binary framing is more appropriate for this dataset

---

