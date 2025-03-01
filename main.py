# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_men)

# Compute correlation matrix
correlation_matrix = train_data.corr(numeric_only=True)

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ['Sex', 'Pclass', 'Embarked', 'Parch'] ##Cannot use fare because of missing value in test data that cannot be manipulated
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

rf_model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
rf_model.fit(X,y)

# Evaluate model
y_pred_rf = rf_model.predict(X)
print("Random Forest Accuracy:", accuracy_score(y, y_pred_rf))
print(classification_report(y, y_pred_rf))

# Standardize only for Logistic Regression (not for RandomForest)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Train logistic regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X, y)

# Evaluate model
y_pred_log_reg = log_reg_model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred_log_reg))
print(classification_report(y, y_pred_log_reg))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)


print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X, y)
y_pred_svm = svm.predict(X)
print("Accuracy:", accuracy_score(y, y_pred_svm))
print(classification_report(y, y_pred_svm))

# Random Forrest had the highest accuracy score so we will submit that one
predictions = rf_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
