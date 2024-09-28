# EmotionDetection-CNN-Optimization-

## Project Overview
This project focuses on **real-time emotion detection** using **Convolutional Neural Networks (CNNs)**, specifically leveraging pre-trained models such as **VGG16** and **ResNet50**. The goal is to build a robust machine learning framework capable of classifying emotions from facial expressions in real time using the **FER2013** dataset. In addition, the project emphasizes **model optimization** by analyzing **learning curves** to fine-tune parameters and enhance generalization performance, ensuring high accuracy on unseen data.

### Key Features:
- **Real-Time Emotion Detection**: Utilizing facial expression recognition to classify emotions into seven categories: anger, disgust, fear, happiness, sadness, surprise, and neutral.
- **Deep Learning Models**: Implementation of three CNN architectures:
  - **Baseline CNN**: A simple custom CNN for benchmarking performance.
  - **VGG16**: A deep learning model pre-trained on ImageNet, fine-tuned for emotion detection.
  - **ResNet50**: A deep residual neural network optimized for complex image classification tasks, addressing vanishing gradient problems.
- **Learning Curve Analysis**: Monitoring training and validation errors across epochs to detect overfitting and underfitting, guiding the optimization process with techniques like **early stopping**.
- **Data Preprocessing and Augmentation**: Techniques like image rescaling, normalization, and augmentation (rotation, flipping, zoom) to prepare the FER2013 dataset for model training.

## Dataset
The project utilizes the **FER2013 dataset**, which contains **35,887 grayscale images**, each with a size of **48x48 pixels**. The images are categorized into seven emotions:
- **Anger**
- **Disgust**
- **Fear**
- **Happiness**
- **Sadness**
- **Surprise**
- **Neutral**

The dataset is split into **28,709 images for training** and **7,178 images for testing**, providing balanced coverage across all emotion categories.

## Key Technologies and Libraries
- **Python 3.8+**
- **TensorFlow/Keras** for deep learning model implementation
- **OpenCV** for image preprocessing
- **Matplotlib/Seaborn** for plotting learning curves and visualizing results

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Siddharthprabhakar/EmotionDetect-CNN-Optimization.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the **FER2013** dataset from [Kaggle](https://www.kaggle.com/msambare/fer2013) and place it in the appropriate directory.
4. Run the training script:
   ```bash
   python train_model.py --model [baseline/vgg16/resnet50] --epochs [number_of_epochs]
   ```
5. View the results and learning curves in the **results/** folder.