
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Prepare data for model training
def data_preparation(filepath):
    df=pd.read_csv(filepath)
    
    # impute all missing values in all the features

    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # drop 'Loan_ID' variable from the data. We won't need it.
    df = df.drop('Loan_ID', axis=1)


    # data type transform to float
    from sklearn.preprocessing import LabelEncoder
    label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Approved']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # handle 'Dependents' 
    df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(int)

 #  One-Hot 'Property_Area'
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['Property_Area']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Property_Area']))
    df = pd.concat([df.drop(columns=['Property_Area']), encoded_df], axis=1)

    # saving this procewssed dataset
    df.to_csv('Processed_Credit_Dataset.csv', index=None)

    # Seperate the input features and target variable
    x = df.drop('Loan_Approved',axis=1)
    y = df.Loan_Approved

    # splitting the data in training and testing set
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=123)

    # scale the data using min-max scalar
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Fit-transform on train data
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    feature_names = xtrain.columns.tolist()  # return feature names
    print(feature_names)
    
    # save encoder 和 scaler
    import pickle
    with open('onehot_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_names

# train model

def Loan_Logistic_Regression(xtrain_scaled, ytrain, xtest_scaled, ytest):
    from sklearn.linear_model import LogisticRegression
    lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)
    
    # Predict the loan eligibility on testing set and calculate its accuracy.
    # First, from sklearn.metrics import accuracy_score and confusion_matrix
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    ypred = lrmodel.predict(xtest_scaled)
    accuracy=accuracy_score(ypred, ytest)
    print('accuracy score is:',accuracy)
    

    # Print the confusion matrix
    cm=confusion_matrix(ytest, ypred)
    print(cm)

    return lrmodel
  

def Loan_Random_Forest(xtrain_scaled, ytrain, xtest_scaled, ytest):
    # Import RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Let's list the tunable hyperparameters for Random Forest algorithm
    RandomForestClassifier().get_params()

    rfmodel = RandomForestClassifier(n_estimators=2,max_depth=2,max_features=10)
    rfmodel.fit(xtrain_scaled, ytrain)

    # predict on xtest
    ypred = rfmodel.predict(xtest_scaled)

    from sklearn.metrics import accuracy_score, confusion_matrix

    print(accuracy_score(ypred, ytest),'\n')
    print(confusion_matrix(ytest, ypred))

    return rfmodel


import pickle

def Loan_cross_validation():
    # import rquired libraries
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # if you have a imbalanced dataset, you can use stratifiedKFold
    from sklearn.model_selection import StratifiedKFold

    #For Logistic Regression cross validation
    # Set up a KFold cross-validation
    kfold = KFold(n_splits=5)

    # Use cross-validation to evaluate the model
    lr_scores = cross_val_score(lrmodel, xtrain_scaled, ytrain, cv=kfold)

    # Print the accuracy scores for each fold
    print("Accuracy scores:", lr_scores)

    # Print the mean accuracy and standard deviation of the model
    print("Mean accuracy:", lr_scores.mean())
    print("Standard deviation:", lr_scores.std())

    #For Random Forest
    # Set up a KFold cross-validation
    kfold = KFold(n_splits=5)

    # Use cross-validation to evaluate the model
    rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)

    # Print the accuracy scores for each fold
    print("Accuracy scores:", rf_scores)

    # Print the mean accuracy and standard deviation of the model
    print("Mean accuracy:", rf_scores.mean())
    print("Standard deviation:", rf_scores.std())
