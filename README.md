# NAC pCR Prediction

Machine learning pipeline for predicting pathologic complete response (pCR) to neoadjuvant chemotherapy (NAC) in breast cancer using MRI-derived radiomic features and deep learning.

## Results

| Model | AUC | F1 Score | Sensitivity | Accuracy |
|-------|-----|----------|-------------|----------|
| Random Forest | **0.728** | **0.615** | **0.794** | **0.745** |

## Tech Stack

- **Machine Learning:** scikit-learn (Random Forest, SVM, XGBoost, Logistic Regression)
- **Deep Learning:** PyTorch (3D CNN)
- **Explainability:** SHAP, GradCAM++
- **Radiomic Features:** PyRadiomics
- **Data Harmonization:** ComBat

## Data Sources (TCIA)

- [QIN-02](https://wiki.cancerimagingarchive.net/display/Public/QIN+Breast+DCE-MRI) — QIN Breast DCE-MRI
- [NACT-Pilot](https://wiki.cancerimagingarchive.net/display/Public/ISPY1) — Neoadjuvant Chemotherapy Pilot
- [I-SPY1](https://wiki.cancerimagingarchive.net/display/Public/ISPY1) — Investigation of Serial Studies to Predict Your Therapeutic Response with Imaging and moLecular Analysis

## Project Structure

```
src/
├── main.py               # Pipeline entry point
├── config.py             # Configuration and hyperparameters
├── data_loader.py        # DICOM/NIfTI loading and preprocessing
├── radiomic_features.py  # 2D radiomic feature extraction
├── radiomic_features_3d.py # 3D radiomic feature extraction
├── harmonize.py          # ComBat batch effect harmonization
├── ml_models.py          # Classical ML models
├── cnn_model.py          # 3D CNN architecture
├── shap_gradcam.py       # SHAP + GradCAM++ explainability
└── visualization.py      # Plots and result visualization

website/
└── index.html            # Interactive results dashboard
```

## Website

Live demo: [nac-pcr-prediction.vercel.app](https://nac-pcr-prediction.vercel.app)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```
