# Predictive Modeling of Cardiovascular Disease

This project focuses on developing and comparing two machine learning models, **Logistic Regression** and **Random Forest**, to predict the risk of cardiovascular disease (CVD) based on clinical and lifestyle factors.

---

## Table of Contents

* [Project Description](#project-description)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Methodology](#methodology)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

---

## Project Description

This research develops and compares two machine learning models—**Logistic Regression** and **Random Forest**—to predict cardiovascular disease (CVD) risk using a dataset of clinical and lifestyle factors.
Through **exploratory data analysis (EDA)** on **68,731 patient records**, key predictors such as **systolic blood pressure, age, and cholesterol** were identified.

---

## Dataset

The dataset originally contained **70,000 records** and **13 features**. After cleaning and preprocessing, the final dataset used for analysis consisted of **68,731 records**.

**Features include:**

* `age`: Age of the patient (in years)
* `gender`: Gender of the patient (Male/Female)
* `height`: Height (cm)
* `weight`: Weight (kg)
* `ap_hi`: Systolic blood pressure
* `ap_lo`: Diastolic blood pressure
* `cholesterol`: Cholesterol level (Normal, Above Normal, Well Above Normal)
* `gluc`: Glucose level (Normal, Above Normal, Well Above Normal)
* `smoke`: Smoking status (Smoker/Non-Smoker)
* `alco`: Alcohol intake (Alcohol/No Alcohol)
* `active`: Physical activity (Active/Inactive)
* `cardio`: Presence of cardiovascular disease (Healthy/At Risk)

---

## Installation

The project was implemented in **Python** with the following libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `shap`

Install dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

---

## Usage

* **Exploratory Data Analysis (EDA):**
  Run the `eda.ipynb` notebook for data cleaning, preprocessing, and visualization.

* **Machine Learning Modeling:**
  Use the `machine_learning.ipynb` notebook for Logistic Regression and Random Forest implementation, including evaluation metrics.

To execute, open Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

---

## Methodology

The workflow includes:

1. **Data Cleaning** – Handling missing values and inconsistencies.
2. **EDA** – Visualizing relationships among features.
3. **Feature Engineering** – Encoding categorical features.
4. **Model Training** – Train-test split (80/20) for Logistic Regression & Random Forest.
5. **Model Evaluation** – Using accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Results

The **Random Forest classifier** outperformed Logistic Regression:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.72     | -         | -      | -        |
| Random Forest       | 0.90     | -         | -      | -        |

---

## Contributing

Contributions are welcome! Please submit a pull request if you’d like to improve the project.

---

## License

This project is licensed under the MIT License.
