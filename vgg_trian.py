from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_preprocessing import test_generator
import numpy as np

# Load the trained model
emotion_model = load_model('emotion_detection_model_vgg.h5')

# Evaluate the model on the test data
test_loss, test_acc = emotion_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Predict the classes for the test set, without limiting steps
Y_pred = emotion_model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)

# True labels from the test set
Y_true = test_generator.classes

# Truncate the predictions to match the number of true labels
Y_pred_classes = np.argmax(Y_pred[:len(Y_true)], axis=1)

# Calculate accuracy score
accuracy = accuracy_score(Y_true, Y_pred_classes)