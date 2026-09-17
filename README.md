# Predicting Kidney Risk from Demographics with Machine Learning

## Objective
This project explores whether demographic variables (age, sex, hospital ward) 
can predict abnormal kidney function (combined Creatinine and BUN thresholds) 
using Logistic Regression, Decision Tree, and Random Forest classifiers.

## Data
354 patients with complete Creatinine (Cr) and BUN values. A combined "kidney 
risk" target was created: at-risk if Cr > 1.5 OR BUN > 23 (standard clinical 
thresholds). 54/354 patients (15%) are flagged as at-risk.

## Method
Models compared: Logistic Regression (age+sex, and age+sex+ward), Decision 
Tree (unrestricted, and depth-limited to max_depth=4), and Random Forest 
(100 trees). Ward was one-hot encoded. Class imbalance was addressed with 
`class_weight='balanced'`. Models were evaluated with a single train/test 
split AND 5-fold stratified cross-validation (shuffled) to test result 
stability.

## Key Findings

| Model | Accuracy | Recall (single split) | Mean Recall (5-fold CV) |
|---|---|---|---|
| Logistic Regression (age+sex) | 0.606 | 0.60 | - |
| Logistic Regression (age+sex+ward) | 0.620 | 0.40 | - |
| Decision Tree (unrestricted) | 0.746 | 0.38 | 0.25 |
| Decision Tree (max_depth=4) | 0.648 | 0.50 | 0.79 |
| Random Forest | 0.803 | 0.25 | 0.24 |

**Cross-validation revealed high variance in recall estimates** (e.g., 0.63 
to 0.94 across folds for the depth-limited tree), driven by the small number 
of at-risk patients (only ~54 total, ~10-11 per fold). This highlights an 
important limitation: with such a small positive class, single train/test 
splits can give misleadingly optimistic or pessimistic results, and 
cross-validation is essential for an honest performance estimate.

## Conclusion
Across all model types, demographic variables (age, sex, ward) showed 
limited and unstable predictive power for kidney risk, compounded by a 
small positive-class sample size. Model complexity did not overcome this 
fundamental data limitation. This is consistent with similar findings in a 
related project on COVID-19 test classification, reinforcing that feature 
relevance and sample size matter more than model sophistication.

## Tools
Python, Pandas, scikit-learn (LogisticRegression, DecisionTreeClassifier, 
RandomForestClassifier, StratifiedKFold, cross_val_score, 
classification_report), Seaborn
