# Logistic Regression for UCI HAR Dataset

# Importing the libraries
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss, roc_curve, auc
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

# Path to the UCI HAR Dataset
train_data_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/train/X_train.txt'
train_labels_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/train/y_train.txt'
test_data_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/test/X_test.txt'
test_labels_path = 'C:/Users/SAHITHI/Desktop/Project/UCI HAR Dataset/test/y_test.txt'

# Importing the training and testing datasets
X_train = pd.read_csv(train_data_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(train_labels_path, delim_whitespace=True, header=None)
X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(test_labels_path, delim_whitespace=True, header=None)

# Combining features and labels into single DataFrames
y_train = y_train.values.ravel()  # Flattening the array to 1D
y_test = y_test.values.ravel()     # Flattening the array to 1D

# Feature Scaling
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating predicted probabilities for loss calculation
y_pred_prob = classifier.predict_proba(X_test)

# Evaluating the model
cm = confusion_matrix(y_test, y_pred)  # Confusion Matrix
ac = accuracy_score(y_test, y_pred)     # Accuracy Score
cr = classification_report(y_test, y_pred)  # Classification Report
loss = log_loss(y_test, y_pred_prob)    # Log Loss

# Calculating bias and variance
bias = classifier.score(X_train, y_train)  # Bias on Training set
variance = classifier.score(X_test, y_test)  # Variance on Test set

# Print evaluation metrics
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", ac)
print("Classification Report:\n", cr)
print("Log Loss:", loss)
print("Bias (Training Accuracy):", bias)
print("Variance (Test Accuracy):", variance)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)  # Assuming label 1 is the positive class
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print current working directory
print("Current Working Directory:", os.getcwd())
