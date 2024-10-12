
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("/Users/krishilparikh/CODING/Compute/compute_task_3/weather_australia.csv")
df.head()

print(df.isnull().sum())

df['MinTemp'].fillna(df['MinTemp'].mean(), inplace=True)
df['MaxTemp'].fillna(df['MaxTemp'].mean(), inplace=True)
df['Rainfall'].fillna(df['Rainfall'].mean(), inplace=True)
df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean(), inplace=True)
df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean(), inplace=True)
df['Temp9am'].fillna(df['Temp9am'].mean(), inplace=True)
df['Temp3pm'].fillna(df['Temp3pm'].mean(), inplace=True)
df['Pressure9am'].fillna(df['Pressure9am'].mean(), inplace=True)
df['Pressure3pm'].fillna(df['Pressure3pm'].mean(), inplace=True)
df['Humidity9am'].fillna(df['Humidity9am'].mean(), inplace=True)
df['Humidity3pm'].fillna(df['Humidity3pm'].mean(), inplace=True)
df.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], inplace=True)
df['WindGustDir'].fillna(df['WindGustDir'].mode()[0], inplace=True)
df['WindDir9am'].fillna(df['WindDir9am'].mode()[0], inplace=True)
df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0], inplace=True)
df['RainToday'].fillna(0, inplace=True)

df.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['WindGustSpeed'], kde=True)
plt.show()
df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean(), inplace=True)
print(df.isnull().sum())

df.dropna(subset=['RainTomorrow'], inplace=True)

df['RainToday'].unique()

df['RainToday'] = df['RainToday'].replace(0 , 'No')

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])

# Extract Month from Date for additional insights
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date', 'Location'], inplace=True)

# Feature scaling
scaler = StandardScaler()
numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall',
                  'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                  'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                  'Temp9am', 'Temp3pm', 'RISK_MM']

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.histplot(df['Rainfall'], kde=True, bins=30)
plt.title('Distribution of Rainfall')
plt.show()

# Distribution of MinTemp
plt.figure(figsize=(12, 6))
sns.histplot(df['MinTemp'], kde=True, color='blue', bins=30)
plt.title('Distribution of Minimum Temperature')
plt.xlabel('Min Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Distribution of MaxTemp
plt.figure(figsize=(12, 6))
sns.histplot(df['MaxTemp'], kde=True, color='orange', bins=30)
plt.title('Distribution of Maximum Temperature')
plt.xlabel('Max Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Distribution of WindSpeed9am
plt.figure(figsize=(12, 6))
sns.histplot(df['WindSpeed9am'], kde=True, color='green', bins=30)
plt.title('Distribution of Wind Speed at 9am')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Frequency')
plt.show()

# Distribution of WindSpeed3pm
plt.figure(figsize=(12, 6))
sns.histplot(df['WindSpeed3pm'], kde=True, color='red', bins=30)
plt.title('Distribution of Wind Speed at 3pm')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Frequency')
plt.show()

X = df.drop(columns=['RainTomorrow'])
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Performance Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(log_reg, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
print(f"Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

import joblib

# Save the model
joblib.dump(log_reg, 'logistic_regression_model.pkl')
print("Model saved as 'logistic_regression_model.pkl'")
