# Loan Default Prediction & Credit Risk Modelling

A comparative machine-learning project for identifying loan applications with elevated default risk using a public loan-default dataset.

## Project overview

- **Dataset:** 148,670 loan applications and 34 variables
- **Target:** `Status` (1 = default, 0 = non-default)
- **Default rate:** 24.64%
- **Modelling features:** 6 selected borrower/loan variables
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting, LightGBM, and Artificial Neural Network (ANN)

The project uses a leakage-safe preprocessing pipeline, validation-based threshold selection, an untouched hold-out test set, and a separate five-fold stratified cross-validation robustness check.

## Selected features

- Upfront charges
- Negative amortisation
- Lump-sum payment
- Property value
- Credit type
- Co-applicant credit type

## Methodology

1. Data quality checks and focused exploratory analysis
2. Stratified 70% train / 15% validation / 15% test split
3. Median imputation and standardisation for numerical variables
4. Most-frequent imputation and one-hot encoding for categorical variables
5. Model training with class-imbalance handling
6. Validation-based threshold optimisation using F1
7. Final evaluation on the untouched test set
8. Five-fold stratified cross-validation with preprocessing and threshold selection repeated inside each fold

## Five-fold cross-validation results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM** | 0.9785 ± 0.0010 | **0.9940 ± 0.0003** | 0.9201 ± 0.0034 | 0.9996 ± 0.0003 | 0.9582 ± 0.0019 | **4.40 ± 0.57** |
| **Gradient Boosting** | **0.9787 ± 0.0008** | 0.9939 ± 0.0002 | **0.9206 ± 0.0030** | **0.9998 ± 0.0002** | **0.9586 ± 0.0016** | 53.28 ± 0.99 |
| Random Forest | 0.9781 ± 0.0008 | 0.9931 ± 0.0003 | 0.9196 ± 0.0029 | 0.9984 ± 0.0009 | 0.9574 ± 0.0014 | 8.81 ± 1.20 |
| ANN | 0.9749 ± 0.0010 | 0.9934 ± 0.0003 | 0.9079 ± 0.0035 | 0.9994 ± 0.0008 | 0.9515 ± 0.0018 | 107.41 ± 25.88 |
| Logistic Regression | 0.8558 ± 0.0018 | 0.7919 ± 0.0045 | 0.8298 ± 0.0074 | 0.5217 ± 0.0049 | 0.6406 ± 0.0045 | 0.35 ± 0.07 |

## Key findings

- **LightGBM achieved the highest mean ROC-AUC: 0.9940 ± 0.0003.**
- **Gradient Boosting achieved the highest mean F1: 0.9586 ± 0.0016.**
- LightGBM produced almost the same F1 while training roughly **12× faster** than Gradient Boosting.
- The ensemble models substantially outperformed Logistic Regression, suggesting that nonlinear relationships and interactions are important in this dataset.
- LightGBM feature importance ranked **property value**, **credit type**, and **co-applicant credit type** as the three most influential feature groups among the six selected variables.

Considering predictive discrimination, stability, and computational efficiency together, **LightGBM is the preferred model for this comparative study**.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, TensorFlow/Keras, Matplotlib, Seaborn

## Repository structure

```text
loan-default-prediction/
├── README.md
├── Loan_Default_Prediction_GitHub_Clean.ipynb
└── requirements.txt
```

The dataset is not included in the repository. To reproduce the analysis, obtain the public Loan Default Dataset (Yasser, 2021) and save it as `Loan_Default.csv` beside the notebook or inside a `data/` folder.

## Limitations

- Results are based on a public historical dataset and may not transfer directly to a live lending portfolio.
- The modelling stage uses six selected features for comparability and interpretability.
- Threshold optimisation maximises F1; a production credit-risk system should incorporate the real financial and regulatory costs of false approvals and false rejections.
- Feature importance reflects predictive association, not causality.

## Future work

Potential extensions include probability calibration, SHAP-based explainability, cross-validated hyperparameter optimisation, fairness testing, and temporal validation/backtesting.
