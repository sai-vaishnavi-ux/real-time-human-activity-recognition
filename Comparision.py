import matplotlib.pyplot as plt

# Sample data for different models
model_names = ['Logistic Regression', 'CNN', 'SVM','AGNN_SAT']  # Replace with your model names
accuracies = [0.95, 0.953, 0.961,0.98]  # Replace with actual accuracy values


# Create a figure and axis
plt.figure(figsize=(5,6 ))

bar_width = 0.1
# Create a bar plot for accuracy
bars = plt.bar(model_names, accuracies, color='blue')

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

# Adding data labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

# Set y-axis limits to make the plot look neat
plt.ylim(0.89, 1.01)

# Show grid lines for better readability
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()