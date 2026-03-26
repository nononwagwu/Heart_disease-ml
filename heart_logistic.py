import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
#testing the new large dataset
import pandas as pd

df = pd.read_csv("heart.csv")

print(df.head())
print(df.shape)
print(df.columns)

df.info()


# column_names = [
#     "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
# ]

# df = pd.read_csv("processed.cleveland.data", names=column_names)

# print(df.head())
# print(df.info())

# import numpy as np

# # Replace ? with NaN
# df.replace("?", np.nan, inplace=True)

# # Convert all columns to numeric
# df = df.apply(pd.to_numeric)

# # Check missing values
# print(df.isnull().sum())

# # remove missing rows
# df.dropna(inplace=True)
# print("Shape after cleaning:", df.shape)

# # convert Target to binary
# df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
# print(df["target"].value_counts())


#Train split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# computing metrics for Logistic Regression
log_accuracy = accuracy_score(y_test, y_pred)
log_precision = precision_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_f1 = f1_score(y_test, y_pred)




#save predictions

pred_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

pred_df.to_csv("logistic_predictions.csv", index=False)
print("Predictions saved.")


# extracting coefficients
import numpy as np

coefficients = model.coef_[0]
features = X.columns

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients,
    "Absolute Weight": np.abs(coefficients)
})

#this will rank features
coef_df = coef_df.sort_values(by="Absolute Weight", ascending=False)
print(coef_df)

#save results
coef_df.to_csv("logistic_feature_importance.csv", index=False)
print("Feature importance saved.")



# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create Model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make Predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the performance

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))


# computing metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)



# save prediction
rf_pred_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": rf_pred
})

rf_pred_df.to_csv("rf_predictions.csv", index=False)
print("RF predictions saved.")

# feature importance

rf_importance = rf_model.feature_importances_

rf_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_importance
})

rf_df = rf_df.sort_values(by="Importance", ascending=False)
print(rf_df)

rf_df.to_csv("rf_feature_importance.csv", index=False)
print("RF feature importance saved.")



# Adding SVM Model
from sklearn.svm import SVC

# Create model
svm_model = SVC()

# Train model
svm_model.fit(X_train, y_train)

# Predictions
svm_pred = svm_model.predict(X_test)


# evaluation

print("\nSVM Results")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))


# compute SVM metrics
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)


# XGBoost model
from xgboost import XGBClassifier

# Create model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# evaluation
print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))

#compute XGBoost metrics
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)




#comparison table

results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "SVM", "XGBoost"],
    "Accuracy": [log_accuracy, rf_accuracy, svm_accuracy, xgb_accuracy],
    "Precision": [log_precision, rf_precision, svm_precision, xgb_precision],
    "Recall": [log_recall, rf_recall, svm_recall, xgb_recall],
    "F1 Score": [log_f1, rf_f1, svm_f1, xgb_f1]
})

# creating comparison table
print(results_df)
results_df.to_csv("model_comparison.csv", index=False)