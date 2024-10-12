# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("/Users/krishilparikh/CODING/Compute/compute_task_4/credit_risk_dataset.csv")

# Display first few rows of the dataset
print(data.head())

# Check missing values
print("Missing values in the dataset:\n", data.isnull().sum())

# Impute missing values
imputer_median = SimpleImputer(strategy='median')
data['person_emp_length'] = imputer_median.fit_transform(data[['person_emp_length']])

imputer_mean = SimpleImputer(strategy='mean')
data['loan_int_rate'] = imputer_mean.fit_transform(data[['loan_int_rate']])

# Confirm missing values are handled
print("Missing values after imputation:\n", data.isnull().sum())

# Encode categorical features using LabelEncoder
label_enc = LabelEncoder()
data['person_home_ownership'] = label_enc.fit_transform(data['person_home_ownership'])
data['loan_intent'] = label_enc.fit_transform(data['loan_intent'])
data['loan_grade'] = label_enc.fit_transform(data['loan_grade'])
data['cb_person_default_on_file'] = label_enc.fit_transform(data['cb_person_default_on_file'])

# Visualization of numerical feature distributions
numerical_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'person_emp_length', 'loan_percent_income', 'cb_person_cred_hist_length']
data[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()

# Bar plots for categorical columns
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i+1)
    sns.countplot(data=data, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Boxplot of numerical features vs loan_status
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=data, x='loan_status', y=col)
    plt.title(f'{col} vs Loan Status')

plt.tight_layout()
plt.show()

# Countplot of categorical features vs loan_status
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i+1)
    sns.countplot(data=data, x=col, hue='loan_status')
    plt.title(f'{col} vs Loan Status')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Prepare data for modeling
X = data.drop(['loan_status'], axis=1)
y = data['loan_status']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVC model
model = SVC()

# Hyperparameter grid for GridSearchCV
param_grid = {
    'C': [0.001, 0.1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1],
}

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters and cross-validation score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Display classification report, confusion matrix, and accuracy
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(best_model, 'svm_credit_risk_model.pkl')
