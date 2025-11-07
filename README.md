# BMW Sales EDA & Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Anastasiya Safonova  
**Date:** 2025  

**Description:** Exploratory Data Analysis of BMW sales (2010–2024) and a classification model to predict sales level (`High` / `Low`).

---

## 📊 Business Context & Problem Statement

Automotive dealers and analysts need to predict whether a BMW model will have **high** or **low** sales in order to optimize inventory, marketing, and financial decisions.  

This project builds a **classification model** using features such as:
**Model**,**Year**,**Region**,**Color**,**Fuel_Type,Transmission**,**Engine_Size_L**,**Mileage_KM**,**Price_USD**,**Sales_Volume** 
to predict the target variable **Sales_Classification** (*High* / *Low*).

The dataset covers **BMW worldwide sales records from 2010 to 2024**,  
including approximately **50,000 records**.

Based on the data:
- **Low sales** correspond to models with `Sales_Volume` between **100–6,999 units**,  
- **High sales** correspond to models with `Sales_Volume` between **7,000–9,999 units**.

This clear boundary in `Sales_Volume` strongly influences the target definition and raises potential concerns about **data leakage**, which the project investigates in detail.


---

## Dataset Overview
- Source: [BMW Worldwide Sales Records (2010–2024) on Kaggle](https://www.kaggle.com/datasets/ahmadrazakashif/bmw-worldwide-sales-records-20102024)  
- ~50,000 records  
- Key features: `Model`, `Year`, `Engine_Size_L`, `Transmission`, `Fuel_Type`, `Color`, `Mileage_KM`, `Price_USD`, `Region`  
- Target variable: `Sales_Classification` (High / Low)

---

## Evaluation Metrics

Since the target variable is **imbalanced**, the model’s performance is evaluated mainly using:

| Metric       | Description |
|-------------|-------------|
| F1 Score     | Harmonic mean of Precision & Recall; balances false positives and false negatives. |
| Precision    | Proportion of predicted `High` sales that are correct. |
| Recall       | Proportion of actual `High` sales correctly identified. |
| ROC AUC      | Ability to distinguish between `High` and `Low` classes. |

**Note:** Accuracy is shown for reference but can be misleading due to class imbalance.

---

## Key Findings
- The target is **imbalanced**, requiring `class_weight='balanced'`.  
- `Sales_Volume` almost perfectly separates classes → **potential data leakage**.  
- Removing `Sales_Volume`, performance drops (Accuracy ~0.57–0.69, ROC AUC ~0.5).  
- Remaining features alone are insufficient for accurate classification, highlighting the importance of feature selection.

---

##  Mini EDA Visualizations

**Target Distribution**
![Class Distribution](\images\sales_class_distribution.png)  

**Sales_Volume vs Sales_Classification (Boxplot)**
![Sales Volume Separation](\images\boxplot_class_volum.png)  

*Observation:*  
`Sales_Volume` shows an almost perfect separation between *Low* and *High* sales classes,  
indicating a potential **data leakage** — a finding later confirmed through modeling.


---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/Safonovanastya87/bmw-sales-eda-classification.git
2. Download the dataset from Kaggle:
   [BMW Worldwide Sales Records (2010–2024)](https://www.kaggle.com/datasets/ahmadrazakashif/bmw-worldwide-sales-records-20102024)
   and place it in the `data/` folder.

3. Install dependencies:
```bash
Копировать код
Next Steps / Recommendations

Explore additional feature engineering.

Test other models (XGBoost, LightGBM).

Apply oversampling/undersampling for imbalanced classes.

Check for dominant features to avoid data leakage.
pip install -r requirements.txt
Launch the notebook:

bash
Копировать код
jupyter notebook notebooks/BMW_Sales_EDA.ipynb
