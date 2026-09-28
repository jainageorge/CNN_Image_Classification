# Building a CNN Model for Image Classification

## 📌 Project Overview

This project focuses on building a **Convolutional Neural Network (CNN)** for image classification using the **Intel Image Classification Dataset**.

The model is trained to classify images into six different categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The project covers the complete image classification workflow, including **data loading, image preprocessing, data augmentation, CNN model building, training, evaluation, and visualization of predictions**.

---

## 🎯 Objectives

The main objectives of this project are:

* Explore the image dataset and identify the different classes.
* Resize images to a common size.
* Normalize pixel values.
* Split the training data into training and validation sets.
* Apply data augmentation to improve model generalization.
* Build a CNN model from scratch using TensorFlow/Keras.
* Train the model using early stopping and model checkpointing.
* Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
* Visualize predictions on random test images.

---

## 📂 Dataset

The project uses the **Intel Image Classification Dataset** available on Kaggle.

The dataset contains images belonging to six classes:

```text
buildings
forest
glacier
mountain
sea
street
```

The dataset is downloaded using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download(
    "puneet6060/intel-image-classification"
)

print("Path to dataset files:", path)
```

---

## 🛠️ Technologies and Libraries

The following technologies and Python libraries are used:

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* KaggleHub

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Explore Dataset
   ↓
Resize Images
   ↓
Normalize Pixel Values
   ↓
Train / Validation Split
   ↓
Data Augmentation
   ↓
Build CNN Model
   ↓
Train Model
   ↓
Early Stopping & Checkpointing
   ↓
Evaluate on Test Data
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Visualize Predictions
```

---

## 🔍 Data Preprocessing

Images are resized to:

```text
150 × 150 pixels
```

The batch size used is:

```text
32
```

Pixel values are normalized from the range `0–255` to `0–1` using:

```python
rescale=1./255
```

### Train-Validation Split

The training dataset is divided into:

* **80% Training**
* **20% Validation**

The test dataset is kept separate and is used only for final evaluation.

---

## 🔄 Data Augmentation

Data augmentation is applied to the training images to improve the model's ability to generalize to unseen images.

The following augmentation techniques are used:

```python
rotation_range=20
width_shift_range=0.10
height_shift_range=0.10
zoom_range=0.20
horizontal_flip=True
```

These transformations create slightly modified versions of training images without changing the original image files.

Validation and test images are only normalized and are **not randomly augmented**.

---

## 🧠 CNN Model Architecture

A CNN model is built from scratch using TensorFlow/Keras.

The architecture contains:

1. Input layer
2. Convolutional layer – 32 filters
3. Max Pooling
4. Convolutional layer – 64 filters
5. Max Pooling
6. Convolutional layer – 128 filters
7. Max Pooling
8. Flatten layer
9. Fully connected Dense layer – 128 neurons
10. Dropout – 0.5
11. Output layer – 6 neurons with Softmax activation

### Model Structure

```text
Input Image
150 × 150 × 3
       ↓
Conv2D (32) + ReLU
       ↓
MaxPooling
       ↓
Conv2D (64) + ReLU
       ↓
MaxPooling
       ↓
Conv2D (128) + ReLU
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense (128) + ReLU
       ↓
Dropout (0.5)
       ↓
Dense (6) + Softmax
       ↓
Predicted Class
```

---

## ⚙️ Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

**Adam** is used to update the model weights during training.

### Loss Function

**Categorical Crossentropy** is used because this is a multi-class classification problem.

### Activation Functions

* **ReLU** is used in the hidden layers.
* **Softmax** is used in the output layer to obtain probabilities for the six classes.

---

## ⏹️ Early Stopping

Early stopping is used to reduce unnecessary training and help prevent overfitting.

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

Training stops when validation loss does not improve for five consecutive epochs.

---

## 💾 Model Checkpointing

The best-performing model is saved using `ModelCheckpoint`.

```python
ModelCheckpoint(
    "best_cnn_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
```

This ensures that the model with the best validation accuracy is retained.

---

## 📊 Model Training

The CNN is trained using the training and validation datasets.

Training and validation accuracy are plotted to understand model performance over epochs.

Training and validation loss are also plotted to identify possible overfitting or underfitting.

---

## 📈 Model Evaluation

The trained CNN is evaluated using the separate test dataset.

The following metrics are calculated:

* Test Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Classification Report

The classification report provides class-wise:

```text
Precision
Recall
F1-score
Support
```

These metrics help understand how well the model performs for each image category.

---

## 🔲 Confusion Matrix

A confusion matrix is used to visualize the actual classes versus the predicted classes.

It helps identify:

* Correct predictions
* Misclassified images
* Classes that are difficult for the model to distinguish

---

## 🖼️ Prediction Visualization

At least 10 random test images are selected and displayed with their:

* Actual class
* Predicted class

This provides a visual understanding of how the CNN performs on unseen images.

---


## 💡 Key Learnings

Through this project, the following concepts were explored:

* Image classification
* Convolutional Neural Networks
* Convolution and feature extraction
* ReLU activation
* Max pooling
* Fully connected layers
* Dropout
* Softmax classification
* Image normalization
* Data augmentation
* Train-validation-test splitting
* Early stopping
* Model checkpointing
* Classification metrics
* Confusion matrix
* Prediction visualization

---

## 📝 Conclusion

This project demonstrates how a **Convolutional Neural Network can be used to classify images into multiple categories**.

The CNN automatically learns visual features from the images through convolutional layers. Data augmentation helps improve generalization by exposing the model to different variations of training images, while dropout and early stopping help reduce overfitting.

The final model is evaluated on unseen test images using accuracy, precision, recall, F1-score, and a confusion matrix. Visualizing random predictions provides an additional way to understand the model's classification performance.

---

## 👩‍💻 Author

**Jaina George**

Data Science & Machine Learning Learner

Skills explored in this project:

**Python | TensorFlow | Keras | NumPy | Matplotlib | Seaborn | Scikit-learn | Machine Learning | Deep Learning | CNN**
