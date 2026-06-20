# Anomaly Detection with MLflow Experiment Tracking

## Project Overview

This project implements an end-to-end fraud detection workflow using MLflow for experiment tracking, model comparison, model registry management, and production model deployment.

The objective is to identify fraudulent transactions from a highly imbalanced dataset and evaluate multiple machine learning models while maintaining a complete ML lifecycle using MLflow.

---

## Dataset

Dataset Source:

https://www.kaggle.com/datasets/kartik2112/fraud-detection

Files Used:

* fraudTrain.csv
* fraudTest.csv

**Note:** Dataset files are not included in this repository due to GitHub file size limitations. Download the dataset from Kaggle and place the CSV files in the project by creating dataset folder before running the notebook.

---

## Technologies Used

* Python 3.11
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* MLflow

---

## Project Workflow

## 1. Data Preprocessing

The following preprocessing steps were performed:

* Loaded fraud detection dataset
* Analyzed class distribution
* Sampled dataset for efficient experimentation
* Removed unnecessary columns
* Encoded categorical variables using LabelEncoder
* Performed stratified train-test split

---

## 2. Handling Class Imbalance

The dataset contains a highly imbalanced fraud class.

SMOTE (Synthetic Minority Oversampling Technique) was applied only on the training dataset.

Purpose:

* Increase minority fraud samples
* Improve fraud detection capability
* Reduce model bias toward majority class

---

## 3. Models Evaluated

### Logistic Regression

Parameters:

* C = 1
* solver = liblinear

---

### Random Forest

Parameters:

* n_estimators = 30
* max_depth = 3

---

### XGBoost

Parameters:

* eval_metric = logloss

---

### XGBoost With SMOTE

Parameters:

* eval_metric = logloss
* SMOTE applied on training data

---

# MLflow Experiment Tracking

## Experiment Name

```
Anomaly Detection MLflow
```

MLflow was used to track all model experiments.

---

## Logged Parameters

* Model hyperparameters
* C
* solver
* n_estimators
* max_depth
* eval_metric
* Sampling strategy

---

## Logged Metrics

* accuracy
* recall_class_0
* recall_class_1
* f1_score_macro

---

## Logged Artifacts

* Trained model artifact
* MLmodel configuration files

---

# Results

| Model               | Accuracy | Recall Class 1 | F1 Macro |
| ------------------- | -------- | -------------- | -------- |
| Logistic Regression | 0.994211  | 0.00000        | 0.498549  |
| Random Forest       | 0.994221  | 0.001776        | 00.500324  |
| XGBoost             | 0.726340  | 0.810835        | 0.930690  |
| XGBoost With SMOTE  | 0.987221  | 0.936945        | 0.726340  |

---

# Best Model

## Selected Model

**XGBoost With SMOTE**

## Reason

Although XGBoost achieved higher accuracy, XGBoost with SMOTE achieved better fraud detection capability by improving recall for fraudulent transactions.

Since anomaly detection focuses on identifying fraud cases, recall of class 1 was considered the most important evaluation metric.

---

# MLflow Model Registry

## Registered Model

```
anomaly-detection-prod
```

## Model Version

```
Version 1
```

## Aliases

```
@challenger
@champion
```

The selected production model was registered in MLflow Model Registry and linked with its source training run.

---

# Production Inference

The production model was loaded from MLflow Model Registry and tested on unseen transaction samples.

Example Output:

```python
[0 0 0 0 0 1 0 0 0 0]
```

---

# Screenshots

The screenshots folder contains:

* MLflow experiment list
* Runs comparison page
* Best model run details
* Model registry view
* Production model details

---

# How to Run

## Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn mlflow
```

---

## Step 2: Start MLflow UI

```bash
mlflow ui --port 5000
```

Open:

```
http://127.0.0.1:5000
```

---

## Step 3: Run Notebook

Execute:

```
anomaly_detection_mlflow.ipynb
```

from beginning to end.

---

# Author

Shikhar Joshi

Project: Anomaly Detection with MLflow Experiment Tracking