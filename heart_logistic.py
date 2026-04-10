import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
#the new large dataset
import matplotlib
print(matplotlib.__file__)

df = pd.read_csv("heart.csv")

print(df.head())
print(df.shape)
print(df.columns)

df.info()
#adding Gridsearch import
from sklearn.model_selection import GridSearchCV


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
# adding scaling for svm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



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

#tuning random forest 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("Best RF Params:", grid_rf.best_params_)

#evaluating the tuned model
best_rf = grid_rf.best_estimator_

rf_tuned_pred = best_rf.predict(X_test)

print("\nTuned Random Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_tuned_pred))
print("Classification Report:\n", classification_report(y_test, rf_tuned_pred))

# store tuned metrics
rf_tuned_accuracy = accuracy_score(y_test, rf_tuned_pred)
rf_tuned_precision = precision_score(y_test, rf_tuned_pred)
rf_tuned_recall = recall_score(y_test, rf_tuned_pred)
rf_tuned_f1 = f1_score(y_test, rf_tuned_pred)




# Adding SVM Model
from sklearn.svm import SVC

# Create model
svm_model = SVC()

# Train model
svm_model.fit(X_train_scaled, y_train)

# Predictions
svm_pred = svm_model.predict(X_test_scaled)


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

# tuning SVM

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_svm.fit(X_train_scaled, y_train)

print("Best SVM Params:", grid_svm.best_params_)

# evaluating tuned SVM
best_svm = grid_svm.best_estimator_

svm_tuned_pred = best_svm.predict(X_test_scaled)

print("\nTuned SVM Results")
print("Accuracy:", accuracy_score(y_test, svm_tuned_pred))
print("Classification Report:\n", classification_report(y_test, svm_tuned_pred))

# store tuned SVM metrics
svm_tuned_accuracy = accuracy_score(y_test, svm_tuned_pred)
svm_tuned_precision = precision_score(y_test, svm_tuned_pred)
svm_tuned_recall = recall_score(y_test, svm_tuned_pred)
svm_tuned_f1 = f1_score(y_test, svm_tuned_pred)


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

#tuning XGBoost


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)

print("Best XGBoost Params:", grid_xgb.best_params_)

# evaluating tuned XGBoost
best_xgb = grid_xgb.best_estimator_

xgb_tuned_pred = best_xgb.predict(X_test)

print("\nTuned XGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_tuned_pred))
print("Classification Report:\n", classification_report(y_test, xgb_tuned_pred))

# store tuned XGBoost metrics
xgb_tuned_accuracy = accuracy_score(y_test, xgb_tuned_pred)
xgb_tuned_precision = precision_score(y_test, xgb_tuned_pred)
xgb_tuned_recall = recall_score(y_test, xgb_tuned_pred)
xgb_tuned_f1 = f1_score(y_test, xgb_tuned_pred)

#creating a graphical comparison of models
svm_before = svm_f1
svm_after = svm_tuned_f1

rf_before = rf_f1
rf_after = rf_tuned_f1

xgb_before = xgb_f1
xgb_after = xgb_tuned_f1
import matplotlib.pyplot as plt
import numpy as np

models = ['SVM', 'Random Forest', 'XGBoost']
before = [svm_before, rf_before, xgb_before]
after = [svm_after, rf_after, xgb_after]

x = np.arange(len(models))
width = 0.35

plt.figure()
plt.bar(x - width/2, before, width, label='Before Tuning')
plt.bar(x + width/2, after, width, label='After Tuning')

plt.xticks(x, models)
plt.ylabel('F1 Score')
plt.title('Hyperparameter Tuning Impact')
plt.legend()

plt.show()
import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

svm_scores = [svm_accuracy, svm_precision, svm_recall, svm_f1]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]
xgb_scores = [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1]

x = np.arange(len(metrics))
width = 0.25

plt.figure()
plt.bar(x - width, svm_scores, width, label='SVM')
plt.bar(x, rf_scores, width, label='Random Forest')
plt.bar(x + width, xgb_scores, width, label='XGBoost')

plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()

plt.show()

import matplotlib.pyplot as plt
import pandas as pd

rf_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(rf_df['Feature'], rf_df['Importance'])
plt.gca().invert_yaxis()

plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')

plt.show()

from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

# Use the tuned SVM
result = permutation_importance(
    best_svm, X_test_scaled, y_test,
    n_repeats=10,
    random_state=42
)

svm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(svm_df['Feature'], svm_df['Importance'])
plt.gca().invert_yaxis()

plt.title('SVM Feature Importance (Permutation)')
plt.xlabel('Importance Score')

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Random Forest importance
rf_imp = rf_model.feature_importances_

# XGBoost importance (USE TUNED MODEL)
xgb_imp = best_xgb.feature_importances_

# SVM importance (already computed earlier)
svm_imp = svm_df.set_index('Feature')['Importance']

# Combine all into one DataFrame
comparison_df = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest': rf_imp,
    'XGBoost': xgb_imp
})

comparison_df = comparison_df.set_index('Feature')

# Add SVM importance (align by feature names)
comparison_df['SVM'] = svm_imp

# Sort by Random Forest importance for clarity
comparison_df = comparison_df.sort_values(by='Random Forest', ascending=False)

# Plot
comparison_df.plot(kind='bar')

plt.title('Feature Importance Comparison Across Models')
plt.ylabel('Importance')

plt.show()