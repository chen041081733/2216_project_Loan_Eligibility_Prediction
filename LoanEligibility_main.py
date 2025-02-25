# Loan_main.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#os.path.join(os.path.dirname(__file__), '..')：返回 src 的上一级目录，sys.path.append(...)：这样就可以让 Python 在 2216 project-part1 目录下查找 src 目录里的模块。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#如果 sys.path.append(...) 写在 import 之后，Python 还是找不到 src，所以一定要在 import 之前添加路径。
from src.Loan_model_select import data_preparation, Loan_Logistic_Regression,Loan_Random_Forest, Loan_cross_validation

file_path = r"C:\algonquin\2025W\2216_ML\2216_project\2216_project_Loan_Eligibility_Prediction\data\credit.csv"

xtrain_scaled, xtest_scaled, ytrain, ytest = data_preparation(file_path)
print(xtrain_scaled[:5])
print(xtest_scaled[:5])
print(ytrain.head())
print(ytest.head())

Loan_Logistic_Regression(xtrain_scaled, ytrain, xtest_scaled, ytest)