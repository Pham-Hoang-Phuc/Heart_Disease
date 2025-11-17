# Heart Disease Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Model-RandomForest-success?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.0%25-orange?style=for-the-badge&logo=none)](https://en.wikipedia.org/wiki/Accuracy_and_precision)
[![Recall](https://img.shields.io/badge/Recall-89.0%25-red?style=for-the-badge&logo=none)](https://en.wikipedia.org/wiki/Precision_and_recall)

This project applies machine learning techniques to predict the presence of heart disease using a clinical dataset of 918 patients with 12 health-related attributes. The goal is to compare multiple supervised models and identify the most effective one for **early disease detection**.

---

## Overview & Motivation

Cardiovascular disease (CVD) remains one of the leading causes of death worldwide. Early and accurate prediction is vital for timely intervention and significantly improving patient outcomes.

In this project, we perform extensive data analysis, extract meaningful clinical insights, and develop robust classification models to predict the target variable, `HeartDisease` (0 = No, 1 = Yes), based on a set of clinical features including: **Age**, **Cholesterol**, **RestingBP**, **MaxHR**, **ST\_Slope**, and **Oldpeak**.

## Dataset & Features

* **Source:** Kaggle – [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
* **Samples:** 918
* **Features:** 12 clinical attributes (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope).
* **Target:** `HeartDisease` (Binary Classification)
---

## Key Insights from Exploratory Data Analysis (EDA)

The EDA phase was crucial for understanding feature distributions and their correlation with heart disease.

| Feature | Correlation Type | Key Finding |
| :--- | :--- | :--- |
| **Oldpeak** | Strong Positive | Higher values are strongly associated with a higher disease risk. |
| **MaxHR** | Strong Negative | Lower maximum heart rates indicate a higher risk profile. |
| **Age** | Moderate Positive | Risk generally increases with age. |
| **Cholesterol & RestingBP** | Weak/Negligible | Showed weak direct correlation with the target variable. |

> **Conclusion:** Features like Oldpeak, MaxHR, and Age are the strongest predictors, suggesting good separability for classification models.

---

## Model Comparison & Evaluation

We trained and evaluated three supervised machine learning models. **Recall** was prioritized as the key metric, given the critical nature of minimizing **False Negatives** (missing a disease case).

| Model | Accuracy | Recall (Prioritized) | AUC | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 86.6% | 84.8% | 0.8702 | Tends to miss more positive cases (lowest Recall). |
| **XGBoost** | 86.6% | 86.0% | 0.8674 | Strong general performance. |
| **Random Forest** | **88.0%** | **89.0%** | **0.8782** | **Best overall performance & highest Recall.** |

### Best Model: Random Forest

The **Random Forest** classifier delivered the highest **Recall (89.0%)** and **F1-score (89.9%)**, making it the most suitable and reliable model for this clinical prediction task.

---

## Tech Stack & Setup

* **Programming Language:** Python 3.8+
* **Core Libraries:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`
* **Modeling:** `scikit-learn`, `XGBoost`
* **Environment:** Jupyter Notebook

### How to Run
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd heart-disease-prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open the main notebook (e.g., `notebooks/HeartDisease_Analysis_Model.ipynb`) and run all cells in order.
---

## Project Structure
````

├── heart.csv/                                \# Raw dataset file
├── heart-disease-prediction-3-models.ipynb/  \# Jupyter notebooks for EDA and Model Training
├── README.md                                 \# Project documentation
└── requirements.txt                          \# Project dependencies
````
