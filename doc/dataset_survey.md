# Dataset Survey for SCG-SNN Project
**Generated:** 2026-05-06
**Purpose:** Identify additional datasets to expand beyond the 19-subject CEBSDB baseline for cross-subject ML generalization.
**Current baseline:** PhysioNet CEBSDB (19 subjects, 58K windows, ODC-BY 1.0)

---

## Table of Contents
1. [Quick Summary Table](#1-quick-summary-table)
2. [Pure SCG Datasets](#2-pure-scg-datasets-detailed)
3. [Multi-modal ECG+SCG Datasets](#3-multi-modal-ecgscg-datasets-detailed)
4. [PCG (Heart Sound) Datasets](#4-pcg-heart-sound-datasets-detailed)
5. [Adjacent-modality Datasets (BCG, large ECG)](#5-adjacent-modality-datasets)
6. [Synthetic Data Tools](#6-synthetic-data-tools)
7. [Top-3 Immediate-Acquire Recommendations](#7-top-3-immediate-acquire-recommendations)
8. [Ranked Table by ROI](#8-ranked-table-by-roi)
9. [Experiments Enabled](#9-experiments-enabled)
10. [Notes on Excluded / Proprietary Sources](#10-notes-on-excluded--proprietary-sources)

---

## 1. Quick Summary Table

| # | Dataset | Modalities | Subjects | Records | fs (Hz) | Format | License | Size | SCG? |
|---|---------|-----------|---------|---------|---------|--------|---------|------|------|
| 1 | **CEBSDB** (current) | ECG+SCG/PCG+RESP | 19 (of 20) | 60 | 5000 | WFDB | ODC-BY 1.0 | ~360 MB | YES |
| 2 | **SCG-GCG-VHD** | ECG+SCG+GCG | 100 | 100 | 256/512 | CSV/MAT/JSON | CC-BY 4.0 | ~981 MB | YES |
| 3 | **SCG-RHC** (UCSF) | ECG+SCG (tri-ax)+RHC | 73 | 83 | 500/1000 | WFDB | ODC-BY 1.0 | unknown | YES |
| 4 | **MSCardio** | SCG (smartphone, tri-ax) | 108 | 502 | variable | CSV/JSON | MIT | ~187 MB | YES |
| 5 | **FOSTER** | FCG+SCG+ECG+PCG+RESP | 40 | 40 | 500-2000 | CSV | CC-BY 4.0 | unknown | YES |
| 6 | **SensSmartTech** | ECG+PCG+PPG+SCG(ACC) | 32 | 338 | 500/1000 | CSV/WFDB | CC-BY-NC-ND 4.0 | ~2.7 MB | YES |
| 7 | **SCG-Pig-Hypovolemia** | SCG (porcine) | 6 pigs | 17059 beats | ~1000 | unknown | unknown | unknown | YES |
| 8 | **Vib2ECG** | SCG(IMU)+12-lead ECG | 17 | multi-day | unknown | unknown | unknown | unknown | YES |
| 9 | **EmoWear** | ECG+ACC/SCG+EDA+RESP | 49 | unknown | 500 | CSV/MAT | CC-BY 4.0 | unknown | YES (IMU) |
| 10 | **CirCor DigiScope** | PCG | 1568 | 5272 | 4000 | WAV/WFDB | ODC-BY 1.0 | ~449 MB | NO |
| 11 | **PhysioNet 2016 PCG** | PCG | ~1072 | 3126 | 2000 | WAV | ODC-BY 1.0 | ~1.1 GB | NO |
| 12 | **PTB-XL** | 12-lead ECG | 18869 | 21799 | 500/100 | WFDB | CC-BY 4.0 | ~1.7 GB | NO |
| 13 | **Chapman-Shaoxing** | 12-lead ECG | 10646 | 10646 | 500 | CSV | ODC-BY 1.0 | ~1.6 GB | NO |
| 14 | **BCG-MultiPathology** | BCG+ECG+Echo | 85 | 153 | 100 | CSV/AVI | CC-BY 4.0 | unknown | NO |
| 15 | **BCG-HeartRhythm** | BCG+3-lead ECG | 46 | 46 overnight | 125/200 | CSV | CC-BY-NC-ND 4.0 | unknown | NO |
| 16 | **BCG-NaturalSleep** | BCG+HR+RESP | 32 | 224 nights | variable | CSV | unknown | unknown | NO |
| 17 | **MIT-BIH Arrhythmia** | 2-ch ECG | 47 | 48 | 360 | WFDB | ODC-BY 1.0 | ~74 MB | NO |
| 18 | **Apnea-ECG** | ECG | 70 | 70 | 100 | WFDB | ODC-BY 1.0 | ~583 MB | NO |
| 19 | **PASCAL HS Challenge** | PCG audio | ~832 | 832 | variable | WAV | public | small | NO |

---

## 2. Pure SCG Datasets (Detailed)

### 2.1 SCG-GCG-VHD -- Valvular Heart Disease Cardio-Mechanical Database

**Source:** Yang C et al., Frontiers in Physiology 2021; doi: 10.3389/fphys.2021.750221
**Repository:** https://doi.org/10.5281/zenodo.5279448

| Field | Value |
|-------|-------|
| Subjects | 100 patients with valvular heart disease (aortic/mitral stenosis) |
| Records | 100 (one per subject; avg 6 min 48 sec) |
| Modalities | ECG (3-channel limb leads) + SCG (3-axis) + GCG (3-axis gyro) |
| Sampling rate | 256 Hz (CP-01..CP-70, UP-01..UP-21) / 512 Hz (UP-22..UP-30) |
| Format | CSV (raw) + MAT (MATLAB) + JSON (fiducial annotations) |
| Annotations | Hand-corrected AO/AC fiducial points + echo params (EF, valve area, mean gradient) |
| License | **CC-BY 4.0** -- free for academic and commercial use with attribution |
| Download size | ~981 MB total (Raw_Recordings.zip 566 MB + MAT_Files.zip 415 MB) |
| Collection sites | Columbia Univ Medical Center + Stevens Institute (USA); Southeast Univ / Nanjing Medical Univ (China) |

**Why useful for this project:**
100 subjects is 5.3x our current 19-subject CEBSDB baseline. The disease labels (VHD vs healthy)
and paired ECG (R-peak ground truth) directly support our label-derivation pipeline. The SCG 3-axis
format matches our CEBSDB channel structure. GCG is a bonus modality for transfer learning.

**Download commands:**
```bash
# Individual files (most reliable through proxy)
wget "https://zenodo.org/records/5279448/files/Raw_Recordings.zip?download=1" -O SCG_GCG_VHD_raw.zip
wget "https://zenodo.org/records/5279448/files/MAT_Files.zip?download=1"      -O SCG_GCG_VHD_mat.zip
wget "https://zenodo.org/records/5279448/files/JSON_Files.zip?download=1"     -O SCG_GCG_VHD_json.zip

# Full archive in one request:
wget "https://zenodo.org/api/records/5279448/files-archive" -O SCG_GCG_VHD_all.zip
```

---

### 2.2 MSCardio -- Mississippi State Remote Cardiovascular Monitoring

**Source:** Taebi and Rahman, Data in Brief 2025; doi: 10.5281/zenodo.14975878
**Repository:** https://zenodo.org/records/14975878
**GitHub:** https://github.com/TaebiLab/MSCardio

| Field | Value |
|-------|-------|
| Subjects | 108 (46 M, 61 F, 1 unspecified; ages 18-62) out of 123 enrolled |
| Records | 502 unique recordings after preprocessing |
| Modalities | SCG only -- tri-axial chest vibrations via smartphone accelerometer |
| Sampling rate | Device-dependent (iOS/Android consumer sensor, variable) |
| Format | CSV (scg.csv per recording) + JSON (metadata) |
| Annotations | Demographic metadata only; no beat-level labels |
| License | **MIT License** |
| Download size | 187 MB (compressed); 9 GB full version |

**Why useful:**
Largest pure-SCG dataset by subject count (108 vs our 19). Smartphone-sourced SCG introduces
domain shift from dedicated sensors -- useful for domain-adaptation and real-world robustness.

**Caution:** No ECG reference, so R-peak-based Sys/Dia labels cannot be derived directly.
Labels must be derived from SCG templates (e.g., matched-filter AO detection).

**Download command:**
```bash
wget "https://zenodo.org/records/14975878/files/TaebiLab/MSCardio-v0.1.zip?download=1" -O MSCardio.zip
```

---

### 2.3 FOSTER -- Forcecardiography Dataset (SCG + ECG + PCG + RESP)

**Source:** Scientific Data 2025; doi: 10.1038/s41597-025-05694-2
**Repository:** https://doi.org/10.17605/OSF.IO/3U6YB

| Field | Value |
|-------|-------|
| Subjects | 40 healthy (20 M, 20 F; mean age 26.93 +/- 7.09 yr) |
| Records | 40 sessions (~7 min each, quiet breathing + apnea phases) |
| Modalities | FCG (broadband force) + SCG + PCG + ECG + RESP |
| Sampling rates | ECG/PCG ~2 kHz; SCG/RESP ~500 Hz |
| Format | CSV |
| Annotations | Quiet breathing vs apnea phases; ECG R-peaks available |
| License | **CC-BY 4.0** |
| Collection | Hypertension Centre, AOU Federico II, Naples, Italy (Jul 2024 - Mar 2025) |

**Why useful:**
Multi-modal ground truth: ECG R-peaks + simultaneous SCG + PCG in one dataset. Enables our
exact label-derivation pipeline (R-peak -> Sys/Dia offset labeling) on 40 additional subjects.
Combined with CEBSDB gives ~59 independent subjects.

**Download command:**
```bash
pip install osfclient
osf -p 3u6yb clone data/foster/
# Or browser download: https://osf.io/3u6yb/
```

---

### 2.4 SCG-RHC -- Wearable SCG + Right Heart Catheter (UCSF / Inan Lab)

**Source:** PhysioNet 2023; DOI: 10.13026/133d-pk11
**Repository:** https://physionet.org/content/scg-rhc-wearable-database/1.0.0/

| Field | Value |
|-------|-------|
| Subjects | 73 patients (14 inpatient HF; 59 outpatient) |
| Records | 83 RHC procedures |
| Modalities | Wearable ECG (1 kHz) + tri-axial SCG (500 Hz) + RHC hemodynamic pressure + ABP + PPG + RESP |
| Sampling rates | Wearable 500 Hz (resampled); RHC 250 Hz |
| Format | WFDB (.hea + .dat) + JSON metadata |
| Annotations | Demographic data + hemodynamic values (CO, SV, RAP, RVP) -- catheter ground truth |
| License | ODC-BY 1.0 |
| Access | **REGIONALLY RESTRICTED** -- not accessible outside USA at time of survey |

**Why useful:**
Gold-standard hemodynamic labels (cardiac output, stroke volume) from simultaneous catheter --
enables regression tasks beyond classification. Heart failure population is clinically relevant.

**Access workaround:** Request via PhysioNet credentialing or contact corresponding author:
Omer T. Inan, Georgia Tech -- omer.inan@ece.gatech.edu

---

### 2.5 SCG-Pig-Hypovolemia -- Porcine SCG with Validated Cardiac Timings

**Source:** Cho MJ et al. (Inan Lab), Scientific Data 2026; doi: 10.1038/s41597-026-06733-2

| Field | Value |
|-------|-------|
| Subjects | 6 porcine subjects |
| Records | 17,059 annotated SCG heartbeats |
| Modalities | SCG + catheter-based pressure (gold-standard AO/AC timing) |
| Annotations | Signal quality index (SQI) + AO/AC fiducial points (inter-annotator verified) |
| License | Not confirmed at survey time |

**Why useful:**
High-quality beat-level annotations with catheter-validated fiducial points. Open-source GUI
annotation tool published alongside -- can be applied to CEBSDB re-annotation.

**Download:** Check paper supplement or contact Inan Lab.
PhysioNet listing not confirmed at time of survey (paper published 2026).

---

### 2.6 SensSmartTech -- Post-Exercise Polycardiographic Dataset

**Source:** Lazovic et al., Scientific Data 2025; DOI: 10.13026/fy9p-n277
**Repository:** https://physionet.org/content/?topic=seismocardiogram (search SensSmartTech)

| Field | Value |
|-------|-------|
| Subjects | 32 (18 F, 14 M; mean age 34.6 +/- 9.1 yr) |
| Records | 338 thirty-second segments (96 pre-exercise, 242 post-treadmill) |
| Modalities | ECG (4-ch, 500 Hz) + SCG/ACC (500 Hz) + PCG (1 kHz) + PPG (100 Hz, 4-ch) |
| Total heartbeats | 17,816 |
| Format | CSV + WFDB |
| Annotations | Exercise protocol phase; ECG R-peaks available |
| License | **CC-BY-NC-ND 4.0** (non-commercial only) |
| Download size | ~2.7 MB |

**Why useful:**
Covers heart rates 52-182 BPM -- critical for stress-testing the classifier at elevated HR.
SCG morphology changes significantly at high HR (Sys/Dia intervals shorten), which is a known
failure mode of fixed-offset labeling.

**Download command:**
```bash
wget -r -N -c -np https://physionet.org/files/senssmarttech/1.0.0/ -P data/senssmarttech/
# Confirm exact PhysioNet slug by searching: https://physionet.org/content/?topic=seismocardiogram
```

---

### 2.7 EmoWear -- Wearable Physiological Dataset (SCG via Chest IMU)

**Source:** Scientific Data 2024; doi: 10.1038/s41597-024-03429-3
**Repository:** https://zenodo.org/records/10407279

| Field | Value |
|-------|-------|
| Subjects | 49 (emotion-elicitation via video clips) |
| Modalities | Chest ACC + GYRO (SCG-equivalent) + ECG + BVP + RESP + EDA + SKT |
| Sampling rate | ECG 500 Hz; ACC 100 Hz |
| Format | CSV + MAT (raw and cleaned packages) |
| License | **CC-BY 4.0** |

**Why useful:**
Dual IMU (accelerometer + gyroscope) on chest, directly comparable to SCG. ECG reference
available. Emotion-elicitation protocol produces varied physiological states (stress, calm)
beyond resting baseline -- useful for evaluating robustness to autonomic arousal.

**Download command:**
```bash
wget "https://zenodo.org/records/10407279/files/EmoWear_raw.zip?download=1" -O EmoWear.zip
```

---

## 3. Multi-modal ECG+SCG Datasets (Detailed)

### 3.1 Vib2ECG -- Paired Multi-Channel SCG+ECG Dataset

**Source:** arXiv:2603.15539 (2026), submitted to IEEE for publication
**Repository (SharePoint):**
https://leidenuniv1-my.sharepoint.com/:f:/g/personal/lug_vuw_leidenuniv_nl/IgDg5j2CIATlQ56qjY4YIf-vAanPDRScblrc056wRheo7Us?e=KZEeAS

| Field | Value |
|-------|-------|
| Subjects | 17 (with multi-day recordings per subject) |
| Modalities | 12-lead ECG + tri-axial SCG/vibrational signals via IMU at 6 chest-lead positions |
| Unique feature | First dataset with chest-lead ECG + SCG paired at same anatomical positions |
| Format | TBD (preprint; awaiting formal publication) |
| License | TBD |

**Why useful:**
The only dataset designed for ECG reconstruction from SCG -- targets whether SCG can replace
chest-lead ECG. Multi-day per-subject recordings enable intra-subject longitudinal analysis.

**Caution:** Small N=17; not yet formally published. Access via SharePoint link above
(research team sharing, may expire). Check arXiv:2603.15539 for latest status.

---

## 4. PCG (Heart Sound) Datasets (Detailed)

### 4.1 PhysioNet/CinC Challenge 2016 -- Heart Sound Classification

**Source:** Clifford et al., CinC 2016
**Repository:** https://physionet.org/content/challenge-2016/1.0.0/

| Field | Value |
|-------|-------|
| Subjects | ~1,072 (healthy + pathological) |
| Records | 3,126 PCG recordings (5-120+ sec each) |
| Modalities | PCG (single-channel) |
| Sampling rate | 2,000 Hz |
| Format | WAV |
| Annotations | Normal / Abnormal binary label |
| License | ODC-BY 1.0 |
| Download size | 1.1 GB uncompressed; ~1,011 MB as ZIP |

**Note:** This is the training.zip that previously failed via proxy. Use AWS S3 or Kaggle mirror
for China-accessible download.

**Download commands:**
```bash
# Option 1 -- wget individual files (more proxy-friendly than single 1 GB zip):
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/ -P data/pcg2016/

# Option 2 -- AWS S3 (no auth, often faster in China than direct PhysioNet):
aws s3 sync --no-sign-request s3://physionet-open/challenge-2016/1.0.0/ ./data/pcg2016/

# Option 3 -- Kaggle community mirror (no PhysioNet auth needed):
pip install kaggle
kaggle datasets download -d bjoernjostein/physionet-challenge-2016
```

**Why useful:**
3,126 recordings for normal/abnormal PCG classification. Pre-training a 1-D CNN encoder on PCG
(same 1-D temporal structure as SCG) then fine-tuning on CEBSDB is a viable domain-transfer path.
Large N compensates for our small 19-subject SCG base.

---

### 4.2 CirCor DigiScope Phonocardiogram Dataset

**Source:** Oliveira et al., IEEE JBHI 2022; George B. Moody PhysioNet Challenge 2022
**Repository:** https://physionet.org/content/circor-heart-sound/1.0.1/

| Field | Value |
|-------|-------|
| Subjects | 1,568 (pediatric; ages 0-21 yr) |
| Records | 5,272 recordings from 4 auscultation locations |
| Modalities | PCG (single-channel) |
| Sampling rate | 4,000 Hz |
| Format | WAV + WFDB headers + TSV segmentation |
| Annotations | Murmur presence/timing/shape/pitch/grade (Levine scale); S1/S2 segmentation |
| License | ODC-BY 1.0 |
| Download size | 449 MB compressed; 558 MB uncompressed |

**Download command:**
```bash
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.1/ -P data/circor/
```

**Why useful:**
Richest PCG annotation in any public dataset. SSL encoder trained on 5,272 PCG recordings
can transfer temporal S1/S2 pattern recognition to SCG Sys/Dia classification. The 1,568-subject
scale provides meaningful pretraining signal relative to our 19-subject fine-tuning set.

---

### 4.3 PASCAL Heart Sound Challenge (2011)

**Source:** Bentley et al. (2011)
**Repository:** http://www.peterjbentley.com/heartchallenge/

| Field | Value |
|-------|-------|
| Records | Dataset A: 176; Dataset B: 656 (total ~832) |
| Classes | A: Normal/Murmur/Extra Heart Sound/Artifact; B: Normal/Murmur/Extrasystole |
| Modalities | PCG audio (WAV) |
| License | Public (no formal license stated) |

**Why useful:**
Historically important baseline. Lower priority than CirCor and PhysioNet 2016 due to smaller
scale. Useful as a quick sanity-check benchmark for PCG classifier heads.

---

## 5. Adjacent-Modality Datasets

### 5.1 PTB-XL -- Large 12-Lead ECG Database

**Source:** Wagner et al., Scientific Data 2020; DOI: 10.13026/x4td-x982
**Repository:** https://physionet.org/content/ptb-xl/1.0.3/

| Field | Value |
|-------|-------|
| Patients | 18,869 |
| Records | 21,799 clinical 12-lead ECG (10 sec each) |
| Sampling rates | 500 Hz (primary) + 100 Hz (downsampled) |
| Format | WFDB (16-bit, 1 uV/LSB) |
| Annotations | 71 SCP-ECG statements: diagnostic (5 superclasses + 24 subclasses) + form + rhythm |
| License | **CC-BY 4.0** |
| Download size | 1.7 GB ZIP; 3.0 GB uncompressed |

**Download command:**
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/ptbxl/
```

**Why useful:**
21K subjects for ECG-only encoder pretraining. A PTB-XL-pretrained encoder can transfer
temporal R-peak pattern priors to SCG. Directly usable as a richer replacement for the
current MIT-BIH + Apnea-ECG component of the SSL mixed corpus (our existing approach).

---

### 5.2 Chapman-Shaoxing 12-Lead ECG Database

**Source:** Zheng et al., 2020
**Repository:** https://physionet.org/content/ecg-arrhythmia/1.0.0/

| Field | Value |
|-------|-------|
| Patients | 10,646 |
| Records | 10,646 (500 Hz, 10 sec each) |
| Format | CSV (5000 rows x 12 columns per file) |
| Annotations | 67 cardiovascular conditions + 11 rhythm labels (expert-labeled) |
| License | ODC-BY 1.0 |
| Download size | ~1.6 GB |

**Download command:**
```bash
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/ -P data/chapman/
```

**Why useful:**
Complementary to PTB-XL; Chinese hospital cohort provides different clinical demographics.
PTB-XL + Chapman together = ~30K unique patients for ECG encoder pretraining.

---

### 5.3 BCG-MultiPathology -- Multi-Pathology Ballistocardiogram

**Source:** Scientific Data 2025; doi: 10.1038/s41597-025-05287-z
**Repository:** https://figshare.com/articles/dataset/28416896

| Field | Value |
|-------|-------|
| Subjects | 85 (sinus rhythm n=67, AF, PVC, PAC, HF n=7) |
| Records | 153 sessions (2-3 per subject) |
| Modalities | BCG + ECG + M-mode echocardiography |
| Sampling rate | 100 Hz |
| Format | CSV + AVI (echo) |
| Annotations | Pathology labels + ejection fraction |
| License | **CC-BY 4.0** |

**Download command:**
```bash
wget "https://figshare.com/ndownloader/articles/28416896/versions/1" -O BCG_MultiPathology.zip
```

**Why useful:**
BCG shares the same physical principle as SCG (body-surface mechanical vibration from cardiac
ejection). Models pretrained on BCG transfer to SCG tasks. 85-subject pathology-labeled dataset
enables disease-classification pretraining.

---

### 5.4 BCG-HeartRhythm -- Overnight Bed BCG with 3-Lead Holter ECG

**Source:** Scientific Data 2025; doi: 10.1038/s41597-025-05936-3
**Repository:** https://doi.org/10.6084/m9.figshare.28643153

| Field | Value |
|-------|-------|
| Subjects | 46 (29 healthy + arrhythmia/ischemia subgroups) |
| Records | 46 overnight sessions (~443 min valid BCG per subject) |
| Modalities | BCG (piezo, 125 Hz) + 3-lead Holter ECG (200 Hz) |
| Format | CSV |
| License | CC-BY-NC-ND 4.0 (non-commercial) |

**Why useful:**
Long-form overnight recordings with simultaneous ECG. Labeled arrhythmia windows (AF, PVC,
ischemia) useful for negative-class augmentation and out-of-distribution testing.

---

### 5.5 BCG-NaturalSleep -- Long-Term Sleep BCG Dataset

**Source:** Scientific Data 2024; doi: 10.1038/s41597-024-03950-5
**Repository:** https://doi.org/10.6084/m9.figshare.26013157

| Field | Value |
|-------|-------|
| Subjects | 32 (7 consecutive nights each in own dormitory) |
| Records | 224 nights total |
| Modalities | BCG (piezo under bedsheet) + reference HR + RESP |
| Format | CSV |
| Collection | Beijing Sport University, Nov 2023 - Jan 2024 |

**Why useful:**
Very long recordings in natural sleep environment. Useful for studying signal drift, body
position changes, and BCG-to-SCG domain shift under realistic conditions.

---

### 5.6 MIT-BIH Arrhythmia Database (already in project)

**Repository:** https://physionet.org/content/mitdb/1.0.0/

| Field | Value |
|-------|-------|
| Subjects | 47 | Records | 48 half-hour 2-channel ECG |
| Sampling rate | 360 Hz | Annotations | Beat-level arrhythmia labels (15 types) |
| License | ODC-BY 1.0 | Download size | ~74 MB |

**Status:** Already downloaded in project (data_mixed/). Used in SSL mixed corpus.

---

### 5.7 Apnea-ECG Database (already in project)

**Repository:** https://physionet.org/content/apnea-ecg/1.0.0/

| Field | Value |
|-------|-------|
| Subjects | 70 (learning 35 + test 35) | Records | 70 (~8 hr each, 100 Hz ECG) |
| Annotations | Per-minute apnea/no-apnea labels | License | ODC-BY 1.0 | Download size | ~583 MB |

**Status:** Already downloaded. ECG-only; low direct relevance to SCG classification.

---

## 6. Synthetic Data Tools

### 6.1 Transformer-Based SCG Synthesis (Inan Lab, 2023)

**Source:** Nikbakht et al., JAMIA 2023; doi: 10.1093/jamia/ocad064
**PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10280352/

- **What it does:** Generates synthetic SCG beats conditioned on AO/AC feature timing and
  participant-specific morphology using a transformer neural network
- **Training data used:** 82 healthy humans (64,894 samples) across 4 internal Georgia Tech datasets
- **Measured benefit:** +3.3% accuracy improvement per 10% synthetic augmentation on downstream classification
- **Code availability:** **NOT PUBLIC** -- "available on reasonable request to corresponding author"
- **Contact:** Omer T. Inan -- omer.inan@ece.gatech.edu

**Why useful:** If code is obtained, augmenting our 58K-window corpus synthetically could
partially substitute for collecting more subjects. The transformer architecture is directly
reproducible in PyTorch from the paper description even without the released code.

---

### 6.2 McSharry ODE ECG/SCG Model

**Source:** McSharry et al., IEEE TBME 2003
**Public implementations:** ecg-kit on GitHub; multiple Python ports

A system of coupled ODEs modeling the ECG waveform. Extended by subsequent work to generate
SCG morphology correlated with simulated ECG. Produces unlimited annotated synthetic windows
with controllable HR, amplitude, and morphology noise.

```python
# Pseudocode outline for synthetic SCG window generation
# from ecgsynth import generate_ecg   # e.g., ecg-kit or neurokit2
# ecg_synth, fs = generate_ecg(N=10000, hr=70, fs=1000, noise_std=0.05)
# r_peaks = detect_r_peaks(ecg_synth, fs)
# scg_synth = generate_scg_morphology(r_peaks, template, noise_level=0.1)
# windows, labels = sliding_window_label(scg_synth, r_peaks, fs)
```

---

### 6.3 GAN-Based ECG/SCG Synthesis

- SLC-GAN (ScienceDirect 2022) and CardioGAN focus on ECG; no confirmed public SCG-GAN
  code repository found as of 2026-05.
- The transformer approach (6.1) is better-validated for SCG specifically.
- For ECG augmentation: `wfdb.processing` provides standard augmentation functions.
- NeuroKit2 (Python) includes synthetic ECG/PPG generation and can be adapted for SCG.

---

## 7. Top-3 Immediate-Acquire Recommendations

### Rank 1 -- SCG-GCG-VHD (Zenodo:5279448) -- ACQUIRE TODAY

**ROI: Highest of all surveyed datasets**

| Criterion | Assessment |
|-----------|-----------|
| Subject count gain | 100 subjects -> 5.3x current 19 |
| Direct SCG | YES -- tri-axial SCG (resample 256->1000 Hz is trivial in scipy) |
| ECG reference | YES -- 3-channel limb leads for R-peak derivation |
| Annotations | Hand-corrected AO/AC fiducial points + echo params |
| License | CC-BY 4.0 -- unrestricted use |
| Download | 981 MB total; no auth; direct wget from Zenodo |
| Integration effort | LOW -- CSV format; existing dataset_pipeline.py needs minor adaptation |
| Clinical value | VHD labels extend beyond healthy-only CEBSDB |

**Exact acquisition commands:**
```bash
mkdir -p data/scg_gcg_vhd
wget "https://zenodo.org/records/5279448/files/Raw_Recordings.zip?download=1" \
     -O data/scg_gcg_vhd/raw.zip
wget "https://zenodo.org/records/5279448/files/JSON_Files.zip?download=1" \
     -O data/scg_gcg_vhd/annot.zip
cd data/scg_gcg_vhd && unzip raw.zip && unzip annot.zip
```

**Experiments unlocked:**
- 5-fold CV on 119 subjects (CEBSDB 19 + VHD 100) -- statistical power up 6x
- Pathology-conditioned classification (healthy vs VHD SCG morphology)
- Fiducial-point regression (AO/AC ground truth now available)
- Cross-dataset domain shift analysis

---

### Rank 2 -- FOSTER Dataset (OSF:3u6yb) -- ACQUIRE TODAY

**ROI: High (direct ECG+SCG multimodal, modern sensors)**

| Criterion | Assessment |
|-----------|-----------|
| Subject count gain | 40 subjects -- doubles cross-subject pool combined with CEBSDB |
| Direct SCG | YES -- simultaneous FCG+SCG+ECG+PCG in one recording session |
| ECG reference | YES -- enables exact same R-peak label pipeline as CEBSDB |
| License | CC-BY 4.0 |
| Download | OSF; no auth; pip install osfclient |
| Integration effort | LOW -- CSV format; same preprocessing pipeline |

**Exact acquisition commands:**
```bash
pip install osfclient
osf -p 3u6yb clone data/foster/
# Or download individual files from: https://osf.io/3u6yb/
```

**Experiments unlocked:**
- CEBSDB (19 subj) + FOSTER (40 subj) = 59 subjects for CV
- Cross-dataset domain shift analysis (hospital sensors vs sleep lab sensors)
- FCG vs SCG signal quality comparison on same subjects

---

### Rank 3 -- CirCor DigiScope PCG (PhysioNet) -- ACQUIRE FOR SSL PRETRAINING

**ROI: High (82x scale-up for pretraining encoder)**

| Criterion | Assessment |
|-----------|-----------|
| Scale | 1,568 subjects, 5,272 recordings -- 82x our SCG dataset |
| Modality | PCG (not SCG) -- domain gap exists but 1-D temporal structure similar |
| Annotations | Richest PCG annotation available (murmur grade, timing, shape) |
| License | ODC-BY 1.0 |
| Download | 449 MB ZIP; global wget; no auth |
| Integration effort | MEDIUM -- needs PCG->SCG encoder adaptation evaluation |

**Exact acquisition commands:**
```bash
# Option A -- recursive wget (proxy-friendlier, resumes on failure):
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.1/ -P data/circor/

# Option B -- single ZIP download:
wget "https://physionet.org/content/circor-heart-sound/get-zip/1.0.1/" -O circor.zip
```

**Experiments unlocked:**
- SSL encoder pretrained on 5,272 PCG recordings, fine-tuned on 58K SCG windows
- Comparison: PCG pretraining vs ECG pretraining vs no pretraining on CEBSDB
- Tests the hypothesis that PCG->SCG transfer beats ECG->SCG (addressing current negative result)
- Murmur detection as auxiliary classification head on SCG windows

---

## 8. Ranked Table by ROI

| Rank | Dataset | SCG? | Subjects | License | Access | Integration | Priority |
|------|---------|------|---------|---------|--------|-------------|----------|
| 1 | **SCG-GCG-VHD** | YES | 100 | CC-BY 4.0 | Free wget (Zenodo) | Low | **ACQUIRE NOW** |
| 2 | **FOSTER** | YES | 40 | CC-BY 4.0 | Free (OSF) | Low | **ACQUIRE NOW** |
| 3 | **CirCor DigiScope** | NO (PCG) | 1568 | ODC-BY | Free wget (PhysioNet) | Medium | Acquire for pretraining |
| 4 | **PhysioNet 2016 PCG** | NO (PCG) | ~1072 | ODC-BY | AWS S3 or Kaggle mirror | Medium | Acquire for pretraining |
| 5 | **SensSmartTech** | YES | 32 | CC-BY-NC-ND | PhysioNet wget | Low | HR stress-test |
| 6 | **PTB-XL** | NO (ECG) | 18869 | CC-BY 4.0 | Free wget (PhysioNet) | Low | ECG pretraining |
| 7 | **MSCardio** | YES | 108 | MIT | Zenodo wget | Medium (no ECG ref) | Domain shift study |
| 8 | **EmoWear** | YES (IMU) | 49 | CC-BY 4.0 | Zenodo wget | Low | Emotion domain test |
| 9 | **BCG-MultiPathology** | NO (BCG) | 85 | CC-BY 4.0 | Figshare | Medium | BCG transfer learning |
| 10 | **SCG-RHC (UCSF)** | YES | 73 | ODC-BY | Restricted (request) | Low | After access granted |

---

## 9. Experiments Enabled

### Experiment A -- Cross-Subject Scale-up (immediate, uses Rank 1+2)

**Datasets:** CEBSDB (19) + SCG-GCG-VHD (100) + FOSTER (40) = 159 subjects

- Re-run 5-fold subject-disjoint CV on 8x more subjects
- Expected: reduce hold-out variance from +/-2.02 pp to ~+/-0.5-1 pp
- New: subject-level failure mode analysis (why do b002/b007 fail? sensor placement? BMI?)
- New: VHD vs healthy SCG morphology -- does the SNN decision boundary generalize?
- New: cross-dataset transfer (train on VHD, test on CEBSDB and vice versa)

### Experiment B -- PCG Pretraining -> SCG Fine-tuning

**Datasets:** CirCor (1568 subj) -> pretrain encoder -> CEBSDB (19 subj) fine-tune

- Tests whether PCG encoder captures S1/S2 mechanical events that map to SCG Sys/Dia
- Hypothesis: PCG->SCG transfer > ECG->SCG (current SSL approach that showed -1.6 pp)
- Directly addresses the finding that SSL mixed corpus hurt accuracy

### Experiment C -- HR Range Robustness

**Datasets:** SensSmartTech (52-182 BPM treadmill range)

- Test SNN classifier on high-HR windows (fixed +/-30 ms Sys offset breaks at >120 BPM)
- Adaptive offset labeling for elevated HR scenarios
- Potential Sys/Dia interval compression model as preprocessing step

### Experiment D -- Smartphone SCG Domain Shift

**Datasets:** MSCardio (smartphone sensors, 108 subjects)

- Train on CEBSDB (dedicated sensor) -> test on MSCardio (consumer sensor)
- Quantifies real-world deployment gap for a wearable consumer product
- Baseline for future consumer-device SCG classifier extension of this FPGA work

---

## 10. Notes on Excluded / Proprietary Sources

| Source | Status | Reason |
|--------|--------|--------|
| Apple Heart Study | Proprietary -- not released | 400K participants; dataset never made public; only summary statistics in NEJM 2019 |
| KAIST SCG Dataset | Not found publicly | KAIST SCG research exists in literature; no confirmed public release as of 2026-05 |
| Tadi / Turku SCG | Not found publicly | J. Tadi (Univ. Turku) published methodology papers; no standalone public dataset identified |
| Inan Lab SCG@home | Not public | Internal Georgia Tech data (64,894 samples across 4 datasets); transformer synthesis paper did not release it |
| Vib2ECG | Partial access | SharePoint link available (arXiv:2603.15539); N=17; awaiting formal publication |
| PASCAL HS Challenge | Superseded | ~832 recordings on personal faculty page; superseded by CirCor and PhysioNet 2016 |
| Michigan HS Library | Education only | 23 exemplar sounds for medical education; not ML-scale dataset |
| SCG Pig Hypovolemia | Location unclear | Described in Scientific Data 2026 (doi: 10.1038/s41597-026-06733-2); PhysioNet/Zenodo location not confirmed at survey time |

---

## Sources and References

All URLs verified May 2026.

- SCG-GCG-VHD Zenodo: https://doi.org/10.5281/zenodo.5279448
- SCG-GCG-VHD paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8519311/
- MSCardio Zenodo: https://zenodo.org/records/14975878
- MSCardio GitHub: https://github.com/TaebiLab/MSCardio
- MSCardio paper (ScienceDirect): https://www.sciencedirect.com/science/article/pii/S2352340925006134
- FOSTER paper (Nature): https://www.nature.com/articles/s41597-025-05694-2
- FOSTER data (OSF): https://doi.org/10.17605/OSF.IO/3U6YB
- SCG-RHC PhysioNet: https://physionet.org/content/scg-rhc-wearable-database/1.0.0/
- SCG-Pig-Hypovolemia (Nature): https://www.nature.com/articles/s41597-026-06733-2
- SensSmartTech (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/
- EmoWear Zenodo: https://zenodo.org/records/10407279
- EmoWear (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11187197/
- Vib2ECG arXiv: https://arxiv.org/abs/2603.15539
- CirCor PhysioNet: https://physionet.org/content/circor-heart-sound/1.0.1/
- CirCor paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC9253493/
- PhysioNet 2016 PCG: https://physionet.org/content/challenge-2016/1.0.0/
- PhysioNet 2016 Kaggle mirror: https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2016
- PASCAL Heart Sound: http://www.peterjbentley.com/heartchallenge/
- PTB-XL PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/
- Chapman-Shaoxing PhysioNet: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- Chapman-Shaoxing Figshare: https://figshare.com/collections/ChapmanECG/4560497/1
- BCG-MultiPathology (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12149309/
- BCG-MultiPathology Figshare: https://figshare.com/articles/dataset/28416896
- BCG-HeartRhythm (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12534543/
- BCG-NaturalSleep (Nature): https://www.nature.com/articles/s41597-024-03950-5
- MIT-BIH: https://physionet.org/content/mitdb/1.0.0/
- Apnea-ECG: https://physionet.org/content/apnea-ecg/1.0.0/
- SCG Transformer Synthesis (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10280352/
- Apple Heart Study (NEJM): https://www.nejm.org/doi/full/10.1056/NEJMoa1901183
- PhysioNet seismocardiogram index: https://physionet.org/content/?topic=seismocardiogram
