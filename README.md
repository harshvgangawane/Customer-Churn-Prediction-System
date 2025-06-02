# Customer Churn Prediction System

## Problem Statement
In today’s highly competitive telecommunications industry, retaining existing customers is as crucial as acquiring new ones. Customer churn when users discontinue using a service leads to significant revenue loss and impacts long-term growth.

Telecom companies must proactively identify customers at risk of churning and implement retention strategies.
## Solution Proposed
This project aims to build a machine learning-based Customer Churn Prediction System that uses historical customer data to predict the likelihood of churn. The goal is to assist business teams in identifying high-risk customers early and taking data-driven decisions to retain them.

## Dataset Used
Dataset link:https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Tech Stack Used
- Python
- Streamlit
- Scikit-learn

## How to Run
**Step 1. Clone the repository.**
```powershell
git clone https://github.com/harshvgangawane/customer-churn-prediction-system.git
cd "Customer Churn Prediction"
```
**Step 2. (Optional) Create a virtual environment.**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**Step 3. Install the requirements**
```powershell
pip install -r requirements.txt
```
**Step 4. Run the Streamlit app**
```powershell
streamlit run app.py
```

## Project Architecture
- Data Layer: CSV files in `notebook/`
- Model Layer: Trained model and scaler in `models/`
- Interface Layer: Streamlit app in `app.py`

## Notebooks
- `ml_pipeline_1.ipynb`: Exploratory Data Analysis
- `ml_pipeline_2.ipynb`: Feature engineering, selection, and modeling

## Models Used
- LogisticRegression (with hyperparameter optimization)
- Feature scaling with MinMaxScaler and PCA
- Label encoding for categorical features

## Conclusion
This project demonstrates a complete ML pipeline for predicting customer churn, from data ingestion to deployment.
The codebase is modular, well-documented, and ready for further extension.
