# Predicting-customer-churn-using-machine-learning-

# ---------------------------------------
# 1. File Upload - Customer Churn CSV
# ---------------------------------------
from google.colab import files
import pandas as pd

uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print(f"\nUploaded file: {filename}")
print(df.head())

# ---------------------------------------
# 2. Data Preprocessing
# ---------------------------------------
print("\nData Info:")
print(df.info())

# Drop duplicates & handle missing values
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ---------------------------------------
# 3. EDA - Churn Distribution
# ---------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Replace with your actual churn column
target_col = 'Churn'  
if target_col not in df.columns:
    raise ValueError("Please replace 'Churn' with your actual target column name.")

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x=target_col)
plt.title("Churn Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# ---------------------------------------
# 4. Feature Engineering
# ---------------------------------------
X = df_encoded.drop(columns=[col for col in df_encoded.columns if 'churn' in col.lower()])
y = df_encoded[[col for col in df_encoded.columns if 'churn' in col.lower()]].iloc[:, 0]

print("\nFeature set shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ---------------------------------------
# 5. Model Building
# ---------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------------------
# 6. Model Evaluation
# ---------------------------------------
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------
# 7. Feature Importance
# ---------------------------------------
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# ---------------------------------------
# 8. Final Output
# ---------------------------------------
print("\n--- FINAL OUTPUT ---")
sample_input = X_test.iloc[0:1]
sample_prediction = model.predict(sample_input)[0]
print("Sample Customer Prediction:", "Will Churn" if sample_prediction == 1 else "Will Not Churn")
