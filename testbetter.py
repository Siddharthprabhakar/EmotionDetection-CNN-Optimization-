import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_preprocessing import test_generator

# Load the trained model
emotion_model = load_model('Epoch-20_vgg.h5')

# Get the class labels
class_labels = list(test_generator.class_indices.keys())

# Predict the classes for the test set
Y_pred = emotion_model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)

# True labels from the test set
Y_true = test_generator.classes

# Convert the predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Get a batch of images from the test set
test_images, test_labels = next(test_generator)

# Visualizing a few test images with predicted and true labels
def plot_images_with_predictions(test_images, Y_true, Y_pred_classes, class_labels):
    plt.figure(figsize=(10, 10))
    for i in range(10):  # Change the range for more/less images
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        true_label = class_labels[Y_true[i]]
        predicted_label = class_labels[Y_pred_classes[i]]
        color = 'green' if true_label == predicted_label else 'red'
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to plot the images
plot_images_with_predictions(test_images, Y_true, Y_pred_classes, class_labels)