# feature selection use rfe

# for data set in df = pd.read_csv(r'D:\heart_attack_prediction_india (2).csv')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif


# Load your dataset
df = pd.read_csv(r'D:\heart_attack_prediction_india (2).csv')

# Drop rows with missing values
df = df.dropna()

# Define target variable (assuming 'Heart Attack Risk' is the dependent variable)
target = 'Heart Attack Risk'
X = df.drop(columns=[target])
y = df[target]

# Initialize logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

# Initialize RFE with the logistic regression model and select 10 features
rfe = RFE(model, n_features_to_select=5)
rfe = rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)
# Initialize SelectKBest with the ANOVA F-value method and select 5 features
select_k_best = SelectKBest(score_func=f_classif, k=5)
X_new = select_k_best.fit_transform(X, y)

# Get the selected features
selected_features = X.columns[select_k_best.get_support()]
print("Selected Features:", selected_features)

for i in range(1, len(X.columns) + 1):
    select_k_best = SelectKBest(score_func=f_classif, k=i)
    X_new = select_k_best.fit_transform(X, y)
    X_selected = X[X.columns[select_k_best.get_support()]]


# Plot accuracy graph
accuracy_scores = []
for i in range(1, len(X.columns) + 1):
    rfe = RFE(model, n_features_to_select=i)
    rfe = rfe.fit(X, y)
    X_selected = X[X.columns[rfe.support_]]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(X.columns) + 1), accuracy_scores, marker='o')
plt.xlabel('Number of Features Selected')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Score vs. Number of Features Selected')
plt.grid(True)
plt.show()