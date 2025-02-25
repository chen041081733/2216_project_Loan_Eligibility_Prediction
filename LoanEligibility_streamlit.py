import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from src.Loan_model_select import data_preparation, Loan_Random_Forest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# cache data upload and model training
@st.cache_data
def load_data_and_model():
    file_path = r"C:\algonquin\2025W\2216_ML\2216_project\2216_project_Loan_Eligibility_Prediction\data\credit.csv"
    try:
        xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_names = data_preparation(file_path)
        logging.info("Data load successfully")
        model = Loan_Random_Forest(xtrain_scaled, ytrain, xtest_scaled, ytest)
        return model, scaler, feature_names
    except Exception as e:
        logging.error(f"Data load fail: {e}")
        st.error("Data load fail, please check again.")
        st.stop()

model, scaler, feature_names = load_data_and_model()

# load coder
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Streamlit interface
st.title("Loan Approval Prediction")

# input
st.sidebar.header("Input infomation for prediction")
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
married = st.sidebar.selectbox('Married', ['Yes', 'No'])
dependents = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.sidebar.selectbox('Self_Employed', ['Yes', 'No'])
applicant_income = st.sidebar.number_input('ApplicantIncome', min_value=0, value=0)
coapplicant_income = st.sidebar.number_input('CoapplicantIncome', min_value=0, value=0)
credit_history = st.sidebar.selectbox('Credit_History', ['1', '0'])
loan_amount = st.sidebar.number_input('LoanAmount', min_value=0, value=0)
loan_term = st.sidebar.selectbox('Loan_Amount_Term', ['360', '180'])
property_area = st.sidebar.selectbox('Property_Area', ['Urban', 'Semiurban', 'Rural'])

# check input data
st.write("original input- ApplicantIncome:", applicant_income)
st.write("original input- CoapplicantIncome:", coapplicant_income)
st.write("original input - LoanAmount:", loan_amount)

# create data frame
input_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'Credit_History': int(credit_history),
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': int(loan_term),
    'Property_Area': property_area
}
input_df = pd.DataFrame(input_data, index=[0])

# data pre processing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed']:
    input_df[col] = le.fit_transform(input_df[col])
input_df['Dependents'] = input_df['Dependents'].replace('3+', '3').astype(int)

# process Property_Area 3 type 
encoded_property = encoder.transform(input_df[['Property_Area']])
encoded_df = pd.DataFrame(encoded_property, columns=encoder.get_feature_names_out(['Property_Area']))
input_df = pd.concat([input_df.drop(columns=['Property_Area']), encoded_df], axis=1)

# align feature names
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# standardize
input_scaled = scaler.transform(input_df)
st.write("data after standardization:", input_scaled)

# prediction
prediction = model.predict(input_scaled)
st.write("Prediction result", prediction)

# show result
if prediction[0] == 1:
    st.sidebar.success("Congratulation, Loan approved")
else:
    st.sidebar.error("Sorry, Loan not approved")
