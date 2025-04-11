Loan_eligibility_application
This app has been built using Streamlit and deployed with Streamlit community cloud

This application predicts whether someone is eligible for a loan based on inputs . The model aims to help users assess loan eligibility by leveraging machine learning predictions.
Features
•	User-friendly interface powered by Streamlit.
•	Input form to enter details such as credit history, loan amount, income, and other relevant factors.
•	Real-time prediction of loan eligibility based on the trained model.
•	Accessible via Streamlit Community Cloud.
Dataset
The application is trained on credit.csv. It includes features like:
•	Gender
•	Marriage
•	Dependents
•	Education
•	Self_Employed
•	Applicant Income
•	Coapplicant Income
•	Credit History
•	Loan Amount
•	Property_Area
Technologies Used
•	Streamlit: For building the web application.
•	Scikit-learn: For model training and evaluation.
•	Pandas and NumPy: For data preprocessing and manipulation.
•	Matplotlib and Seaborn: For exploratory data analysis and visualization (if applicable).
Model
The predictive model applies preprocessing steps like encoding categorical variables and scaling numerical features. The classification model used may include algorithms such as Logistic Regression, Random Forest, or XGBoost.

Installation (for local deployment)
If you want to run the application locally, follow these steps:
1.	Clone the repository:
2.	git clone https://github.com/chen041081733/2216_project_Loan_Eligibility_Prediction.git
cd 2216_project_Loan_Eligibility_Prediction
3.	Create and activate a virtual environment:
python -m venv env
source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
4.	Install dependencies:
pip install -r requirements.txt
5.	Run the Streamlit application:
streamlit run LoanEligibility_Streamlit.py

Thank you for using the Credit Eligibility Application!  

Streamlit Link:
https://2216projectloaneligibilityprediction-bwneebb7xq2sjycefdeyz5.streamlit.app/


