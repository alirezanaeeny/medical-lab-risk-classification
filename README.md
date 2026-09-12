# medical-lab-risk-classification

## Overview
This project analyzes a hospital laboratory dataset (`BIOCBC.csv`, Persian-language lab export) that combines **CBC (Complete Blood Count)** parameters with **biochemistry panel** results (glucose, lipid profile, kidney function, liver enzymes, electrolytes, HbA1c, CRP, RF). The goal is to preprocess raw multilingual/mixed-format lab data and build simple **binary classification models** to flag patients at risk for common clinical conditions based on age, sex, and hospital ward.

## Data Preprocessing
- Loaded a semicolon-separated CSV encoded in `cp1256` (Windows Arabic/Persian encoding).
- Renamed Persian/lab-code column headers into clean English names (e.g. `hb`, `hct`, `rbc`, `wbc`, `plt`, `fbs`, `cr`, `bun`, `sgot`, `sgpt`, `chol`, `tg`, `hdl`, `ldl`, `na`, `k`, `a1c`, `crp`, `rf`).
- Cleaned the `age` column by stripping Persian suffixes (`ساله` = "years old", `روزه` = "days old") and converting to numeric.
- Cleaned the `ward` column by removing the Persian word for "ward" (`بخش`).
- Translated `sex` values from Persian (`مرد`/`زن`) to English (`male`/`female`).
- Built four task-specific sub-datasets by dropping rows with missing values relevant to each clinical question:
  - **Diabetes**: `fbs`, `sex`, `ward`, `age`
  - **Lipid profile**: `tg`, `chol`, `ward`, `age`, `sex`
  - **Kidney function**: `cr`, `bun`, `ward`, `age`, `sex`
  - **Liver function**: `sgot`, `sgpt`, `ward`, `age`, `sex`

## Labeling (Clinical Thresholds)
Continuous lab values were converted into binary risk labels using standard reference cutoffs:

| Marker | Threshold (abnormal = 1) |
|---|---|
| FBS (fasting blood sugar) | > 120 |
| Triglycerides | > 160 |
| Cholesterol | > 240 |
| Creatinine | > 1.5 |
| BUN | > 23 |
| SGOT (AST) | > 40 |
| SGPT (ALT) | ≥ 40 |
| Liver risk (combined) | SGOT abnormal **or** SGPT abnormal |

## Models
Four logistic regression classifiers (`class_weight="balanced"` to handle class imbalance) were trained with an 80/20 train-test split (`random_state=42`):

1. **Creatinine risk** — features: `age`, `sex`
2. **Creatinine risk with SMOTE** — oversampling the minority class on the training set to compare against the baseline
3. **Creatinine risk with ward dummies** — features: `age`, `sex`, one-hot encoded `ward`
4. **BUN risk** — features: `age`, `sex`, one-hot encoded `ward`
5. **Liver risk (combined SGOT/SGPT)** — features: `age`, `sex`, one-hot encoded `ward`

Each model reports accuracy, a confusion matrix, and a full classification report, visualized with a Seaborn heatmap.

## Results Summary

| Model | Accuracy | Notes |
|---|---|---|
| Creatinine (age, sex) | 60.6% | Very low recall/precision on the abnormal class (minority class) |
| Creatinine + SMOTE | 60.6% | Oversampling did not meaningfully change test performance |
| Creatinine + ward dummies | 62.0% | Slight improvement; convergence warning (needs more iterations/scaling) |
| BUN + ward dummies | 67.6% | Best of the kidney models, but abnormal-class precision still very low |
| Liver (SGOT/SGPT combined) | 68.4% | Best balanced precision/recall, but tiny test set (n=19) |

**Key takeaway:** Accuracy looks moderate, but the models struggle badly on the minority ("abnormal") class — precision for the positive class is very low across the board (e.g. 0.08–0.10 for creatinine). This is a classic **class imbalance problem** made worse by a **very small sample size** after dropping missing values (e.g. only 71 test rows for kidney, 19 for liver). SMOTE on such a small dataset did not help. Results should be interpreted as a learning exercise, not a clinically valid model.

## Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
seaborn
matplotlib
scipy
openpyxl
```

## How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib scipy openpyxl
python biocbc_disease_risk_prediction.py
```
Place `BIOCBC.csv` in the same directory as the script.

## Limitations & Next Steps
- Sample sizes per sub-task are small (20–354 rows), which limits reliable model evaluation.
- Severe class imbalance in every target variable; SMOTE alone was not sufficient.
- Only demographic features (age, sex, ward) were used as predictors — no lab-derived features were included as inputs (they were only used to build the labels), so predictive power is inherently limited.
- The `max_iter` convergence warning on logistic regression suggests features should be scaled (e.g. `StandardScaler`) and/or `max_iter` increased.
- Future work: feature scaling, cross-validation instead of a single train/test split, trying tree-based models (Random Forest, XGBoost) which handle imbalance and mixed features better, and using proper imbalance metrics (ROC-AUC, PR-AUC, balanced accuracy) instead of relying on plain accuracy.
