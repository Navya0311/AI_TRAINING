# mlr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r'D:\heart_attack_prediction_india (2).csv')


# Drop rows with missing values
df = df.dropna()
# 'Gender', 'Fasting Blood Sugar (> 120 mg/dL)', 'Resting ECG Results',
#        'Slope of Peak Exercise ST Segment', 'Smoking History',
#        'Obesity (BMI > 30)', 'Hypertension History', 'Diabetes History',
#        'Physical Activity', 'Stress Levels'
# selected features


# Define target variable (assuming 'Category' is the dependent variable)
target = 'Heart Attack Risk'
X = df[['Gender', 'Resting ECG Results', 'Obesity (BMI > 30)',
       'Hypertension History', 'Physical Activity']]
y = df[target]
print(X.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate model
y_pred = model.predict(X_test)
# display graph of logistic regression

print("Roc Curve", model.predict_proba(X_test))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print("Classification Report:\n", classification_report(y_test, y_pred))
