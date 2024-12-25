import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib
#whatever things I need to import
jobs_df = pd.read_csv("resume_dataset.csv") #connect excel file
jobs_df.columns # read feature by feature
encoder = OrdinalEncoder()
jobs_df[["Education"]] = encoder.fit_transform(jobs_df[["Education"]])
jobs_df[["Awards"]] = encoder.fit_transform(jobs_df[["Awards"]])
jobs_df[["Job Role"]] = encoder.fit_transform(jobs_df[["Job Role"]])
jobs_df[["Previous Jobs"]] = encoder.fit_transform(jobs_df[["Previous Jobs"]])
#encode the diff features into numbers the computer can understand
jobs_df.describe() #generate descriptive statistics
X = jobs_df[['Education', 'Awards', 'Previous Jobs']]
Y = jobs_df['Accepted or Rejected']
# which is features and which is objective
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#which for test which for train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Standardize the data
model = LogisticRegression()
#initialize model of training
model.fit(X_train, y_train)
#train model on data
y_pred = model.predict(X_test_scaled)
#Make predictions based on test data
accuracy = accuracy_score(y_test, y_pred)
#calculate accuracy
print("Original X_train:")
print(X_train)
print("\nScaled X_train:")
print(X_train)
#training results
print("\nOriginal X_test:")
print(X_test)
print("\nScaled X_test:")
print(X_test_scaled)
#testing results
print("\nPredictions on X_test:")
print(y_pred)
#Y predicions
print("\nActual y_test:")
print(y_test.values)
#results
print("\nAccuracy of the model:")
print(accuracy)
#accuracy
joblib.dump(model, 'hiring.joblib')
joblib.dump(scaler, 'scaler.joblib')
#save my model















