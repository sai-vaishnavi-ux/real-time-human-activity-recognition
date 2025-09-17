# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Path to the UCI HAR Dataset
train_data_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/train/X_train.txt'
train_labels_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/train/y_train.txt'
test_data_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/test/X_test.txt'
test_labels_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/test/y_test.txt'

# Load the training and test sets
X_train = pd.read_csv(train_data_path, delim_whitespace=True, header=None).values
y_train = pd.read_csv(train_labels_path, delim_whitespace=True, header=None).values.ravel()
X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None).values
y_test = pd.read_csv(test_labels_path, delim_whitespace=True, header=None).values.ravel()

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Bias and Variance
bias = classifier.score(X_train, y_train)
print("Training Score (Bias):", bias)
variance = classifier.score(X_test, y_test)
print("Test Score (Variance):", variance)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Cross-validation accuracy
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')

# Plotting the Cross-Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cv_scores, label='Cross-Validation Accuracy', marker='o', linestyle='--')
plt.title("Cross-Validation Accuracy vs Folds")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Print the mean and standard deviation of cross-validation scores
print("Cross-validation Mean Accuracy: ", cv_scores.mean())
print("Cross-validation Std Deviation: ", cv_scores.std())