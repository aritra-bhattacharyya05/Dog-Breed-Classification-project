# 🐶 Dog Breed Classifier

A deep learning–based web application that identifies dog breeds from images using a convolutional neural network (CNN). The model leverages **transfer learning** to achieve high accuracy across 120 dog breed classes and is deployed as an interactive web app.

---

## 🔗 Live Demo & Notebook

* 🌐 **Deployed App**: https://dog-breed-classification-project-lnvuhvb2miyamovrbo7d7x.streamlit.app/
* 📓 **Kaggle Notebook**: https://www.kaggle.com/code/aribhzz/dog-vision

---

## 📌 Overview

This project demonstrates a complete machine learning pipeline — from data preprocessing and model development to real-world deployment.

The system takes an image of a dog and predicts its breed along with confidence scores. It is designed to handle diverse real-world conditions such as lighting variations, background noise, and pose differences.

---

## 🧠 Model Architecture

The model is built using **Convolutional Neural Networks (CNNs)** with a **transfer learning** approach.

### Key concepts used:

* **CNN (Convolutional Neural Network)** for feature extraction
* **Transfer Learning** to reuse pretrained knowledge
* Fine-tuning for domain-specific learning

### Architecture details:

* **Base Model**: MobileNetV2 (pretrained on ImageNet)
* **Custom Layers**:

  * Global pooling layer
  * Dense layers
  * Dropout (regularization)
* **Output Layer**: Softmax (120 classes)
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy

The pretrained network captures general visual patterns, while fine-tuning adapts it specifically for dog breed classification.

---

## 📊 Dataset

* **Total Classes**: 120 dog breeds
* **Dataset Size**: ~10,000 images
* **Source**: Kaggle Dog Breed Identification dataset

### Preprocessing:

* Resized images to **224 × 224**
* Normalized pixel values
* Applied data augmentation:

  * Rotation
  * Flipping
  * Zooming

These steps improve robustness and generalization.

---

## ⚙️ Training Strategy

* Transfer learning with staged fine-tuning
* Early stopping to avoid overfitting
* Dropout for regularization
* Efficient batch training

The model learns discriminative features such as fur texture, ear structure, and facial characteristics.

---

## 📈 Results

* **Validation Accuracy: 95.5%**
* **Top-5 Accuracy: Very high**, capturing visually similar breeds effectively

👉 A **95.5% accuracy** on a **120-class classification problem** reflects strong model performance and effective use of CNNs with transfer learning.

### Model strengths:

* High accuracy on clear, single-dog images
* Strong generalization across many breeds
* Meaningful top-5 predictions for similar breeds

### Limitations:

* Lower confidence on:

  * Mixed breeds
  * Blurry or low-resolution images
  * Occluded subjects

---

## 🚀 Deployment

The model is deployed using **Streamlit**, providing an interactive interface where users can:

* Upload an image
* View predicted breed
* See confidence scores
* Explore top-5 predictions

### Engineering challenges addressed:

* TensorFlow dependency conflicts
* Keras 2 vs Keras 3 compatibility
* TensorFlow Hub integration
* Cloud deployment debugging

---

## 🧪 Example Output

* **Predicted Breed**: Golden Retriever
* **Confidence**: 92.4%

**Top Predictions:**

1. Golden Retriever — 92.4%
2. Labrador Retriever — 4.1%
3. Flat-Coated Retriever — 2.3%

---

## 💡 Key Learnings

* Practical implementation of **CNNs and transfer learning**
* Handling large-scale multi-class classification
* Debugging real-world deployment issues
* Building end-to-end ML applications

---

## 🏁 Conclusion

This project demonstrates how **CNNs combined with transfer learning** can achieve high accuracy (95.5%) in complex image classification tasks. It showcases both strong model performance and the ability to deploy machine learning systems in a real-world environment.

---
