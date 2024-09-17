# 🩺 Medical Imaging Classifier

A comprehensive deep learning project for classifying medical images, leveraging **TensorFlow**, **Transfer Learning**, **Scikit-learn**, **NumPy**, and **OpenCV**. This project explores advanced CNN architectures, hyperparameter optimization, and robust evaluation to achieve a top accuracy of **88%** in medical image classification. 🏆

---

## ✨ Project Overview

- **Objective**: Develop a high-performing classifier for medical images using Transfer Learning and CNN architectures.
- **Technologies Used**:
  - **TensorFlow**: For building and training deep learning models.
  - **Transfer Learning**: Fine-tuned pre-trained CNN models.
  - **Scikit-learn**: For hyperparameter tuning (Grid Search) and evaluation metrics.
  - **NumPy** & **OpenCV**: For image preprocessing and data augmentation.
  - **Matplotlib**: For visualizing model performance.
- **Timeline**: Aug 2024 – Sep 2024

---

## 🧠 Key Achievements

- 🔍 **Model Exploration**: Fine-tuned five CNN architectures: **DenseNet201**, **ResNet50**, **MobileNetV2**, **VGG19**, and **InceptionV3** on high-dimensional medical imaging datasets.
- ⚙️ **Hyperparameter Tuning**: Optimized learning rates (0.1, 0.01, 0.001), batch sizes, and optimizers (**Adam**, **SGD**, **RMSprop**) using Grid Search, improving performance by **20%**.
- 📉 **Overfitting Mitigation**: Implemented dropout layers and data augmentation (e.g., rotation, flipping) to reduce overfitting.
- 📊 **Comprehensive Evaluation**: Evaluated models across multiple metrics: **Accuracy**, **Precision**, **Recall**, **Cohen’s Kappa**, and **Jaccard Index**, averaging results over 9 combinations per model (3 optimizers × 3 learning rates).
- ✅ **Top Performance**: Achieved a classification accuracy of **88%**, with detailed performance visualizations.

---

## 🛠️ Methodology

### 1. Data Preprocessing
- Used **OpenCV** and **NumPy** to preprocess medical images (resizing, normalization).
- Applied data augmentation techniques (rotation, flipping, scaling) to increase dataset diversity and reduce overfitting.

### 2. Model Training
- Fine-tuned pre-trained CNN models (**DenseNet201**, **InceptionV3**, **MobileNetV2**, **ResNet50**, **VGG19**) using **TensorFlow** and **Transfer Learning**.
- Trained models with various optimizers (Adam, SGD, RMSprop) and learning rates (0.1, 0.01, 0.001).

### 3. Hyperparameter Optimization
- Employed **Grid Search** (via Scikit-learn) to tune:
  - Learning rates: 0.1, 0.01, 0.001
  - Batch sizes
  - Optimizers: Adam, SGD, RMSprop
- Resulted in a **20% performance improvement**.

### 4. Model Evaluation
- Evaluated models on multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - Cohen’s Kappa (inter-rater agreement)
  - Jaccard Index (similarity between predicted and true labels)
- Computed average metrics across 9 configurations per model (3 optimizers × 3 learning rates).

## 🚀 Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Meetjain-0201/Medical-Imaging-Classifier.git
   cd Medical-Imaging-Classifier