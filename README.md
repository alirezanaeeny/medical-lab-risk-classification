# Predicting Kidney Risk from Demographics with Machine Learning

## Objective
This project explores whether demographic variables (age, sex, hospital ward) 
can predict abnormal kidney function (combined Creatinine and BUN thresholds) 
using Logistic Regression and Decision Tree classifiers.

## Data
Laboratory records filtered to 354 patients with complete Creatinine (Cr) and 
BUN values. A combined "kidney risk" target was created: a patient is flagged 
as at-risk if either Cr > 1.5 or BUN > 23 (standard clinical thresholds).

- At-risk: 54 (Cr) / 59 (BUN), combined into a single binary target
- Ward was one-hot encoded to include hospital department as a feature

## Method
Three models were compared:
1. Logistic Regression (age + sex)
2. Logistic Regression (age + sex + ward)
3. Decision Tree Classifier (age + sex + ward), with and without depth limiting

Class imbalance was addressed using `class_weight='balanced'` and, separately, 
SMOTE oversampling. Models were evaluated with Accuracy, Confusion Matrix, and 
Precision/Recall/F1-score — accuracy alone is misleading on imbalanced medical data.

## Key Findings

| Model | Accuracy | Recall (at-risk) | F1-score |
|---|---|---|---|
| Logistic Regression (age+sex) | 0.606 | 0.60 | 0.18 |
| Logistic Regression (age+sex+ward) | 0.620 | 0.40 | - |
| Decision Tree (unrestricted) | 0.746 | 0.38 | 0.25 |
| Decision Tree (max_depth=4) | 0.648 | 0.50 | 0.24 |

The unrestricted Decision Tree showed signs of overfitting (high accuracy but 
low recall); limiting tree depth improved recall at the cost of accuracy. 
Across all model types, demographic variables alone remained insufficient 
predictors of kidney risk — consistent with similar findings in a related 
project on COVID-19 test classification.

## Conclusion
Demographic data (age, sex, ward) has limited predictive power for lab test 
abnormalities without clinically relevant biomarkers. Model complexity 
(Decision Tree vs. Logistic Regression) did not overcome this fundamental 
data limitation — highlighting that feature relevance matters more than 
model choice.

## Tools
Python, Pandas, scikit-learn (LogisticRegression, DecisionTreeClassifier, 
train_test_split, classification_report), imbalanced-learn (SMOTE), Seaborn
