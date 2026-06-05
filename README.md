# Handwritten Digit Recognition using Machine Learning

This repository features a machine learning-based system capable of recognizing handwritten numerical digits (0-9) from grayscale images using supervised learning techniques. The project explores data preprocessing, evaluates classification models, and provides an interactive interface for real-time predictions.

---

## 👥 Authors & Credits
This project was collaboratively developed by:
* **Taha Shakeel** — Dawood University of Engineering and Technology (DUET) 
* **Muskan Irfan** — University of Management & Technology (UMT)

---

## 📌 Project Overview
Handwritten digit recognition is a core computer vision task with applications ranging from postal mail sorting to bank cheque processing. The primary goal of this project was to train and contrast distance-based algorithms against neural networks to determine the most effective approach for classification.

### Key Goals:
* Train machine learning models on labeled digit images.
* Evaluate model performance on unseen data.
* Compare K-Nearest Neighbors (KNN) and Multi-Layer Perceptron (MLP) architectures.
* Provide an interactive user interface for live testing.

---

## 📊 Dataset Description
The model utilizes a dataset consisting of $28\times28$ pixel grayscale images:
* **Total Features:** 784 flattened pixel columns.
* **Target Labels:** Numerical digits from 0 to 9.
* **Data Structure:** The first column acts as the `label` (digit), while the remaining columns represent individual pixel intensity values (0-255).

---

## 🛠️ Methodology & Implementation

### 1. Preprocessing
Pixel values were normalized from `[0, 255]` to a scale of `[0, 1]` to accelerate model convergence and optimize training performance. 

### 2. Models Evaluated
* **K-Nearest Neighbors (KNN):** A simple, distance-based instance classifier configured with $k=3$.
* **Multi-Layer Perceptron (MLP):** A type of neural network capable of learning complex patterns, configured with a hidden layer size of $(128,)$.

### 3. Quick Code Reference
```python
# Normalizing data
X_train = train_df.iloc[:, 1:].values / 255.0
X_test = test_df.iloc[:, 1:].values / 255.0

# Fitting KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Fitting Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10)
mlp.fit(X_train, y_train)
