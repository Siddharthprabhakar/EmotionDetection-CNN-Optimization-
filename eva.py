import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data logged during training
log_data = pd.read_csv('vgg_training.log')

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))

plt.plot(log_data['epoch'], log_data['accuracy'], label='Training Accuracy')
plt.plot(log_data['epoch'], log_data['val_accuracy'], label='Validation Accuracy')

# Add labels and title
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Display the plot
plt.grid(True)
plt.show()