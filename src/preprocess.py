import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/credit_risk_dataset.csv")

# Step 1: filling missing values in the specific columns

median_emp_length = df["person_emp_length"].median()
df["person_emp_length"] = df["person_emp_length"].fillna(median_emp_length)

mean_int_rate = df["loan_int_rate"].mean()
df["loan_int_rate"] = df["loan_int_rate"].fillna(mean_int_rate)

# Step 2: handling outliers : cap the outliers in person_emp_length.
df["person_emp_length"] = df["person_emp_length"].clip(
    upper=60
)  # clip(upper=60) means anything above this 60 replace it with 60


# Step 3: encoding categorical columns
# print(df.dtypes)   checking which are not numeric to encode it
# print(df.select_dtypes(include='str').columns)


df = pd.get_dummies(
    data=df,
    columns=[
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ],
    drop_first=True,
)


# Step 4 : split the data
# X = all features value except "loan_status"
# y = target variable

X = df.drop(columns="loan_status")
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
