import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Use raw string (r'...') or double backslashes to avoid path issues
df = pd.read_csv(r'C:\Users\AK\OneDrive\Desktop\ml\Project 4\Social_Network_Ads.csv')

# Display the first few rows
print(df.head())

# Step 2: Select features and target variable
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Fit Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = classifier.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Step 7: Function to visualize decision boundaries
def plot_decision_boundary(X_set, y_set, title, classifier, scaler):
    X1, X2 = X_set[:, 0], X_set[:, 1]
    X1_grid, X2_grid = np.meshgrid(
        np.arange(start=X1.min() - 1, stop=X1.max() + 1, step=0.01),
        np.arange(start=X2.min() - 1, stop=X2.max() + 1, step=0.01)
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(X1_grid, X2_grid,
                 classifier.predict(np.array([X1_grid.ravel(), X2_grid.ravel()]).T).reshape(X1_grid.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1_grid.min(), X1_grid.max())
    plt.ylim(X2_grid.min(), X2_grid.max())

    # Plot actual points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X1[y_set == j], X2[y_set == j],
                    color=ListedColormap(('red', 'green'))(i), label=f'Class {j}')

    plt.title(title)
    plt.xlabel('Age (Standardized)')
    plt.ylabel('Estimated Salary (Standardized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Step 8: Visualize Training set
plot_decision_boundary(X_train, y_train, 'Naive Bayes (Training Set)', classifier, scaler)

# Step 9: Visualize Test set
plot_decision_boundary(X_test, y_test, 'Naive Bayes (Test Set)', classifier, scaler)

y_pred = classifier.predict(X_test)


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nPrecision Score:", precision_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

df['SalaryPerAge'] = df['EstimatedSalary'] / (df['Age'] + 1)
X = df[['Age', 'EstimatedSalary', 'SalaryPerAge']].values
# Try different smoothing levels
classifier = GaussianNB(var_smoothing=1e-9)
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

#balancing the dataset
print(df['Purchased'].value_counts())

#cross validation 
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-validated Accuracy:", scores.mean())
