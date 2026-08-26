# Loan Default Prediction & Credit Risk Modelling

A comparative machine-learning study of five supervised models for predicting loan default risk using a public lending dataset.

## Project overview

- **Dataset:** 148,670 loan applications and 34 original variables
- **Target:** `Status` (`1 = Default`, `0 = Non-default`)
- **Default rate:** approximately 24.6%
- **Final predictors:** 6 borrower/loan variables selected through the original correlation-based screening process
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, LightGBM, and Artificial Neural Network (ANN)
- **Evaluation:** 70/30 hold-out split using Accuracy, ROC-AUC, F1-score, classification reports, confusion matrices, ROC curves, and feature importance

This repository presents a portfolio-ready reproduction of the modelling workflow used in the original study. The analytical logic and reported results are intentionally preserved for consistency with the study.

## Selected predictors

- `Upfront_charges`
- `Neg_ammortization`
- `lump_sum_payment`
- `property_value`
- `credit_type`
- `co-applicant_credit_type`

## Reported results

| Model | Threshold | Accuracy | ROC-AUC | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7500 | 0.8779 | 0.9487 | 0.7729 |
| Random Forest | 0.6313 | 0.8908 | 0.9630 | 0.7929 |
| Gradient Boosting | 0.6896 | 0.8997 | 0.9666 | 0.8022 |
| **LightGBM** | **0.6831** | **0.9001** | **0.9672** | **0.8030** |
| ANN | 0.4000 | 0.8982 | 0.9662 | 0.8002 |

**LightGBM achieved the strongest overall result** in the original comparison, with Accuracy of **0.9001**, ROC-AUC of **0.9672**, and F1-score of **0.8030**.

## Workflow

1. Load and inspect the loan-default dataset
2. Explore class balance and selected borrower/loan variables
3. Remove exact duplicate rows
4. Apply the original missing-value treatment and label encoding
5. Select the six final predictors
6. Standardise the feature matrix
7. Split the dataset into 70% training and 30% testing
8. Train and compare five supervised-learning models
9. Evaluate using threshold-based classification metrics, ROC curves, confusion matrices, and feature importance

## Visual analysis included

The notebook contains:

- target class distribution
- categorical predictor comparisons
- numerical predictor distributions
- compact correlation matrix
- model performance comparison
- ROC curves for all five models
- confusion matrices for all five models
- feature-importance comparison
- ANN training history

## Repository structure

```text
loan-default-prediction/
├── README.md
├── loan_default_prediction_portfolio.ipynb
├── requirements.txt
└── .gitignore
```

The dataset is not stored in this repository. To reproduce the analysis, obtain the public **Loan Default Dataset** by Yasser (2021) and save it as `Loan_Default.csv` beside the notebook or inside a `data/` folder.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, TensorFlow/Keras, Matplotlib, Seaborn

## Methodology note

This repository reproduces the original study rather than retrospectively changing its analytical design. In the original implementation, feature selection and standardisation were performed before the train/test split, and the reported probability thresholds were retained from the original F1-optimisation stage.

For a production-grade credit-risk system, a stricter workflow would fit preprocessing and feature selection only on training data and use a separate validation set or nested cross-validation for threshold selection.

## Limitations

- The analysis uses a public dataset rather than an institutional lending portfolio.
- The original study uses a single 70/30 hold-out split rather than repeated cross-validation.
- The final comparison is based on six selected predictors.
- ANN training can vary slightly between runs due to stochastic optimisation.
- Predictive feature importance should not be interpreted as causal impact.

## Future work

Potential extensions include train-only preprocessing, k-fold cross-validation, probability calibration, SHAP-based explainability, temporal backtesting, fairness testing, and explicit leakage auditing.
