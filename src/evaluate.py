"""
evaluate.py
-----------
 Evaluating the model to check if its perform well or not
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = joblib.load("models/loan_model.pkl")
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

y_pred = model.predict(X_test)

# Checking Accuracy ,recall,precision, f1score

accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1_Score = f1_score(y_test,y_pred)


print("Accuracy:", accuracy)
print("Recall:",recall)
print("precision:",precision)
print("f1 Score:",f1_Score)