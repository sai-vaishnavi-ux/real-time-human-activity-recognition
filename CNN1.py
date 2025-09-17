# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# Reshape data for CNN (assuming 561 features in each instance, reshape to (561, 1) for Conv1D)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train - 1)  # Subtracting 1 to start labels from 0
y_test = to_categorical(y_test - 1)

# Building the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model in .h5 format
model.save('C:/Users/SAHITHI/Desktop/Project/models/human_activity_model.h5')

# Load the model for evaluation or later use
model = load_model('C:/Users/SAHITHI/Desktop/Project/models/human_activity_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict the Test set results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Activity labels
activity_labels = {
    0: 'Walking',
    1: 'Walking Upstairs',
    2: 'Walking Downstairs',
    3: 'Sitting',
    4: 'Standing',
    5: 'Laying'
}

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:\n", cm)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[activity_labels[i] for i in range(6)], yticklabels=[activity_labels[i] for i in range(6)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
cr = classification_report(y_test_classes, y_pred_classes)
print("Classification Report:\n", cr)

# Plotting the Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

