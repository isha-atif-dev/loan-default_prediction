"""
train.py
--------
This script trains three machine learning models on the loan default dataset:
- Logistic Regression
- Decision Tree
- Random Forest

The best model will be saved to models/loan_model.pkl for use in evaluation.
"""

import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# loading files
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# creating the model
model1 = RandomForestClassifier(class_weight='balanced')
model2 = LogisticRegression(class_weight='balanced',max_iter=1000)
model3 = DecisionTreeClassifier(class_weight='balanced')

# traning the models
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)

# predicting the data
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)


# compare the three models

print("RandomForest accuracy:",accuracy_score(y_test,y_pred1))
print("LogisticRegression accuracy:",accuracy_score(y_test,y_pred2))
print("DecisionTree accuracy:",accuracy_score(y_test,y_pred3))

# saving the best model among three to churn_model.pkl
# by using the library joblib

joblib.dump(model1,"models/loan_model.pkl")
print("model saved succesfully")
