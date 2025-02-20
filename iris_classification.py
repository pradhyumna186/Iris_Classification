import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['target_names'] = df['target'].map(target_names)

# Exploratory Data Analysis
print(df.describe())
print(df.info())

# Visualizations
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='target_names')
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(by='target_names', figsize=(10, 6))
plt.show()

# Prepare data for model training
X = df.drop(['target', 'target_names'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC())
]

# Train and evaluate classifiers
results = []
for name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((name, accuracy))
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("\n")

# Select the best classifier
best_classifier = max(results, key=lambda x: x[1])
print(f"The best classifier is {best_classifier[0]} with an accuracy of {best_classifier[1]:.4f}")
