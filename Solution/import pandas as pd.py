import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("G:/Heart Disease prediction/Data/heart.csv")  # Make sure your dataset file is named 'heart.csv' and is in the same directory

# Data Preprocessing
# Handle missing values (if any)
df=df.fillna(df.mean())

# Normalize continuous features
scaler = StandardScaler()
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Convert categorical variables into numerical formats using one-hot encoding
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal', 'ca'], drop_first=True)

# Split the dataset into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Hyperparameter Tuning
# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': range(1, 21), 'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(knn, knn_params, cv=10, scoring='accuracy')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_random = RandomizedSearchCV(rf, rf_params, n_iter=100, cv=10, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, cm

# Evaluate KNN
knn_results = evaluate_model(best_knn, X_test, y_test)
print("KNN Results:")
print(f"Accuracy: {knn_results[0]:.2f}")
print(f"Precision: {knn_results[1]:.2f}")
print(f"Recall: {knn_results[2]:.2f}")
print(f"F1 Score: {knn_results[3]:.2f}")
print(f"ROC AUC: {knn_results[4]:.2f}")
print(f"Confusion Matrix:\n{knn_results[5]}")

# Evaluate Random Forest
rf_results = evaluate_model(best_rf, X_test, y_test)
print("\nRandom Forest Results:")
print(f"Accuracy: {rf_results[0]:.2f}")
print(f"Precision: {rf_results[1]:.2f}")
print(f"Recall: {rf_results[2]:.2f}")
print(f"F1 Score: {rf_results[3]:.2f}")
print(f"ROC AUC: {rf_results[4]:.2f}")
print(f"Confusion Matrix:\n{rf_results[5]}")

# Feature Importance for Random Forest
feature_importances = best_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest')
plt.gca().invert_yaxis()
plt.show()

# SHAP Analysis
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type='bar')
shap.summary_plot(shap_values[1], X_test)

# LIME Analysis
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),
                                                   feature_names=X.columns,
                                                   class_names=['No Disease', 'Disease'],
                                                   mode='classification')

i = 25  # Example index from the test set
exp = explainer.explain_instance(data_row=X_test.iloc[i], predict_fn=best_rf.predict_proba)
exp.show_in_notebook(show_table=True)
