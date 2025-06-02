import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

model_path=os.path.join('models', 'logistic_regression.pkl')
model=joblib.load(model_path)
pca_path=os.path.join('models', 'pca.pkl')
pca=joblib.load(pca_path)
scaler_path = os.path.join('models', 'minmax_scaler.pkl')
minmax = joblib.load(scaler_path)
label_path=os.path.join('models','labelencoder.pkl')
le=joblib.load(label_path)
st.title('Customer Churn Prediction System')

def user_input_features():
    gender=st.selectbox('gender',['Male','Female'])
    Partner=st.selectbox('Partner',['Yes','No'])
    PhoneService=st.selectbox('Phone Service',['Yes','No'])
    MultipleLines=st.selectbox('Multiple Lines',['No','Yes','No phone service'])
    InternetService=st.selectbox('Internet Service',['No','Fiber optic','DSL'])
    OnlineSecurity=st.selectbox('Online Security',['No','Yes','No internet service'])
    OnlineBackup=st.selectbox('Online Backup',['No','Yes','No internet service'])
    DeviceProtection=st.selectbox('Device Protection',['No','Yes','No internet service'])
    TechSupport=st.selectbox('Tech Support',['No','Yes','No internet service'])
    StreamingTV=st.selectbox('Streaming TV',['No','Yes','No internet service'])
    StreamingMovies=st.selectbox('Streaming Movies',['No','Yes','No internet service'])
    PaperlessBilling=st.selectbox('Paperless Billing',['Yes','No'])
    PaymentMethod=st.selectbox('Payment Method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
    MonthlyCharges=st.number_input('Monthly Charges')
    TotalCharges=st.number_input('Total Charges')
    tenure_group=st.selectbox('Tenure Group',['0-1 year','1-2 years','2-3 years','3-4 years','4-5 years','5+years'])
    is_long_term_contract=st.selectbox('Is Long Term Contract',[0,1],format_func=lambda x:'No' if x==0 else 'Yes')
    senior_with_dependents=st.selectbox('Senior with Dependents',[0,1],format_func=lambda x:'No' if x==0 else 'Yes')
    
    data={
        'gender':gender,
        'Partner':Partner,
        'PhoneService':PhoneService,
        'MultipleLines':MultipleLines,
        'InternetService':InternetService,
        'OnlineSecurity':OnlineSecurity,
        'OnlineBackup':OnlineBackup,
        'DeviceProtection':DeviceProtection,
        'TechSupport':TechSupport,
        'StreamingTV':StreamingTV,
        'StreamingMovies':StreamingMovies,
        'PaperlessBilling':PaperlessBilling,
        'PaymentMethod':PaymentMethod,
        'MonthlyCharges':MonthlyCharges,
        'TotalCharges':TotalCharges,
        'tenure_group':tenure_group,
        'is_long_term_contract':is_long_term_contract,
        'senior_with_dependents':senior_with_dependents
    }
    features=pd.DataFrame(data,index=[0])
    
    return features

def preprocess_input(df):
    df['tenure_group'] = le.transform(df['tenure_group'])
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
    df['InternetService'] = df['InternetService'].map({'No': 0, 'Fiber optic': 1, 'DSL': 2})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['OnlineBackup'] = df['OnlineBackup'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['DeviceProtection'] = df['DeviceProtection'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['TechSupport'] = df['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['StreamingTV'] = df['StreamingTV'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['StreamingMovies'] = df['StreamingMovies'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    df['is_long_term_contract'] = df['is_long_term_contract'].astype(int)
    df['senior_with_dependents'] = df['senior_with_dependents'].astype(int)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'])
    df[['MonthlyCharges', 'TotalCharges']] = minmax.transform(df[['MonthlyCharges', 'TotalCharges']])
    expected_cols = [
        'gender', 'Partner', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'tenure_group',
        'is_long_term_contract', 'senior_with_dependents'
    ]
    df = df[expected_cols]
    df = df.astype(float)
    df_pca = pca.transform(df)

    return df_pca

input_features=user_input_features()

if st.button('Predict'):
    processed_features = preprocess_input(input_features)
    prediction = model.predict(processed_features)
    prediction_proba = model.predict_proba(processed_features)[0][1]
    st.subheader('Prediction result')
    if prediction == 1:
        st.error(f'This customer is likely to churn. Probability: {prediction_proba:.2%}')
    else:
        st.success(f'This customer is not likely to churn. Probability: {1 - prediction_proba:.2%}')
    
uploaded_file=st.sidebar.file_uploader("Upload your file",type=['csv'])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.subheader('Dataset Preview')
    st.write(df.head())
    
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Churn', ax=ax)
    ax.set_xticklabels(['No', 'Yes'])
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Tenure Group")
    tenure_churn = df.groupby('tenure_group')['Churn'].mean().sort_index()
    st.bar_chart(tenure_churn)

