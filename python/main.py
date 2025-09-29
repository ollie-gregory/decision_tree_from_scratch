import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

# Data preprocessing
data = pd.read_csv('../loan_data.csv')
data.drop(columns=['Loan_ID'], inplace=True)
data = data.dropna()

# replace '3+' with '3' in dependents column
data['Dependents'] = data['Dependents'].replace('3+', '3')
data['Dependents'] = data['Dependents'].astype(int)


# From scratch implementation
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"From scratch accuracy: {accuracy * 100:.2f}%")

# Sklearn implementation for comparison
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sklearn_tree = SklearnDecisionTree()
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)

accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn accuracy: {accuracy_sklearn * 100:.2f}%")