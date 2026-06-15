# 🧠 Brain Tumor MRI Classification Using Deep Learning

## 📌 Project Overview

Brain tumors are among the most critical neurological disorders requiring timely diagnosis and treatment. Manual interpretation of Magnetic Resonance Imaging (MRI) scans is often time-consuming and dependent on expert radiologists. This project presents an automated Brain Tumor MRI Classification System using Deep Learning techniques to classify MRI images into four categories:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

The project compares the performance of a Custom Convolutional Neural Network (CNN) with a Transfer Learning approach using EfficientNetB0. The best-performing model is then deployed using Streamlit to provide an interactive web application for real-time MRI image classification.

---

## 🎯 Objectives

* Develop an automated system for brain tumor classification using MRI images.
* Perform Exploratory Data Analysis (EDA) to understand the dataset.
* Build and evaluate a Custom CNN model from scratch.
* Implement Transfer Learning using EfficientNetB0.
* Compare both models using multiple evaluation metrics.
* Deploy the final model using Streamlit for real-time predictions.

---

## 📂 Dataset Information

The dataset consists of brain MRI scans categorized into four classes:

| Class      | Description                          |
| ---------- | ------------------------------------ |
| Glioma     | Tumors originating from glial cells  |
| Meningioma | Tumors arising from the meninges     |
| No Tumor   | MRI scans without any tumor          |
| Pituitary  | Tumors affecting the pituitary gland |

### Dataset Split

| Dataset        | Number of Images |
| -------------- | ---------------- |
| Training Set   | 1,695            |
| Validation Set | 502              |
| Testing Set    | 246              |

### Class Distribution (Training Set)

| Class      | Images |
| ---------- | ------ |
| Glioma     | 564    |
| Meningioma | 358    |
| No Tumor   | 335    |
| Pituitary  | 438    |

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Class distribution analysis
* Sample MRI image visualization
* Image resolution consistency check
* Class distribution pie chart

Generated visualizations include:

* Class Distribution Pie Chart
* Sample MRI Images

---

## ⚙️ Data Preprocessing

The MRI images were preprocessed using the following techniques:

* Image resizing to 224 × 224 pixels
* Pixel normalization
* Data augmentation techniques:

  * Random Horizontal Flip
  * Random Rotation
  * Random Zoom

These preprocessing steps improve model generalization and reduce overfitting.

---

## 🧠 Models Implemented

### 1. Custom CNN

A Convolutional Neural Network was built from scratch using TensorFlow/Keras.

#### Architecture

* Data Augmentation Layer
* Rescaling Layer
* Conv2D (32 Filters)
* Batch Normalization
* MaxPooling
* Conv2D (64 Filters)
* Batch Normalization
* MaxPooling
* Conv2D (128 Filters)
* Batch Normalization
* MaxPooling
* Global Average Pooling
* Dense Layer (256 Units)
* Dropout (0.5)
* Output Layer (Softmax)

#### Techniques Used

* Early Stopping
* Model Checkpointing
* Dropout Regularization

---

### 2. Transfer Learning – EfficientNetB0

EfficientNetB0 pretrained on ImageNet was used for transfer learning.

#### Methodology

* Load EfficientNetB0 without the classification head
* Freeze pretrained layers initially
* Add custom classification layers
* Train the classifier head
* Fine-tune the top layers
* Save the best-performing model

---

## 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

## 📈 Results

### Custom CNN

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 72.76% |
| Precision     | 77%    |
| Recall        | 73%    |
| F1 Score      | 72%    |

Although the Custom CNN achieved satisfactory performance, it exhibited instability across multiple runs and occasionally converged to predicting a single dominant class.

---

### EfficientNetB0

| Metric        | Value      |
| ------------- | ---------- |
| Test Accuracy | **86.59%** |
| Precision     | **86.85%** |
| Recall        | **86.59%** |
| F1 Score      | **86.07%** |

EfficientNetB0 demonstrated superior and more stable performance compared to the Custom CNN.

---

## 🏆 Model Comparison

| Metric    | Custom CNN | EfficientNetB0 |
| --------- | ---------- | -------------- |
| Accuracy  | 72.76%     | 86.59%         |
| Precision | 77%        | 86.85%         |
| Recall    | 73%        | 86.59%         |
| F1 Score  | 72%        | 86.07%         |

### Final Selected Model

**EfficientNetB0** was selected for deployment due to its higher accuracy, robustness, and better generalization performance.

---

## 💻 Streamlit Deployment

The final Streamlit application provides:

### 🔍 MRI Prediction Module

* Upload MRI images (JPG, JPEG, PNG)
* Display uploaded MRI scan
* Predict tumor type
* Show confidence score
* Display probability distribution for all classes

### 📊 Project Analytics Dashboard

* Class Distribution Pie Chart
* Training Curves
* Confusion Matrices
* Class-wise Metrics Heatmap
* Model Comparison Chart
* Metrics Table

### 📚 About Project

* Project Objective
* Models Used
* Final Selected Model
* Disclaimer

---

## 📁 Project Structure

```text
brain-tumor-classification/
│
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── custom_cnn_final.h5
│   ├── efficientnet_final.h5
│   └── class_names.npy
│
├── results/
│   ├── class_distribution_pie_chart.png
│   ├── model_comparison.png
│   ├── metrics.csv
│   ├── custom_cnn_confusion_matrix.png
│   ├── efficientnet_confusion_matrix.png
│   ├── custom_cnn_training_curves.png
│   ├── efficientnet_training_curves.png
│   └── classwise_metrics_heatmap.png
│
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* PIL
* Streamlit

---

## ⚠️ Disclaimer

This application is intended solely for educational and research purposes. The predictions generated by this system should not be considered a substitute for professional medical advice, diagnosis, or treatment. Clinical decisions should always be made by qualified healthcare professionals.

---

## 👨‍💻 Author

**Avi Sood**

Brain Tumor MRI Classification using Deep Learning and Transfer Learning with EfficientNetB0.
