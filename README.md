# 🍽️ Food Detection System Using YOLO and Streamlit

This repository contains an end-to-end **multi-class food object detection system** built using **YOLO (Ultralytics)** and deployed via a **Streamlit web application**.  
The project covers the full pipeline from **raw dataset preprocessing** to **model training, evaluation, and deployment for inference**.

---

## 📌 Project Motivation

Automatic food recognition is an important component in applications such as:
- dietary monitoring,
- nutrition analysis,
- smart food logging systems, and
- assistive technologies.

Manual food annotation is time-consuming and error-prone. This project aims to leverage **deep learning–based object detection** to automatically detect and classify multiple food items in a single image.

---

## 🎯 Objectives

- Develop a **multi-class food detection model** using YOLO
- Perform **systematic dataset preprocessing and cleaning**
- Train and evaluate the model using standard object detection metrics
- Deploy the trained model in an **interactive web application**
- Provide a reproducible and well-documented pipeline

---

## 🧠 Methodology Overview

The overall workflow of the project is shown below:

1. Raw dataset collection (UECFood256)
2. Data inspection and cleaning
3. Annotation format conversion (Pascal VOC → YOLO)
4. Dataset splitting (train / validation / test)
5. Model training and evaluation
6. Deployment for real-time inference

---

## 📊 Dataset

### Source
- **UECFood256 Dataset**
- Original annotation format: **Pascal VOC (XML)**

### Dataset Issues Addressed
- Missing labels
- Missing or corrupted images
- Duplicate samples
- Inconsistent class indexing

### Dataset Stages

Due to large file upload issues, use the following link to download the raw dataset: 
https://binusianorg-my.sharepoint.com/personal/nicholas_victorio_binus_ac_id/_layouts/15/guestaccess.aspx?share=IgA7Q6ah7klrSYZXQSfG43TPAQtcaT7a1LOBqFARY7cd22g&e=GGZUaO

| Folder | Description |
|------|------------|
| `raw/` | Original dataset (unchanged) |
| `dataset/` | Temporary working copy for preprocessing |
| `dataset_raw/` | Frozen early-stage snapshot |
| `dataset_working/` | Cleaned and converted dataset |
| `dataset_final/` | Final YOLO dataset (train/val/test) |

---

## ⚙️ Data Preprocessing

Preprocessing steps include:
- Validation of image–label pairs
- Removal of corrupted and duplicate files
- Conversion from Pascal VOC (XML) to YOLO format
- Normalization of class indices
- Train / validation / test split

The final dataset follows the YOLO directory structure and is used exclusively for model training and evaluation.

---

## 🤖 Model Architecture

- Model family: **YOLO (Ultralytics)**
- Variant: **YOLO11s**
- Task: **Multi-class object detection**
- Input size: `640 × 640`

YOLO is chosen for its balance between **accuracy and real-time inference speed**, making it suitable for interactive applications.

---

## 🏋️ Training Setup

- Framework: Ultralytics YOLO
- Optimizer: AdamW
- Loss: YOLO detection loss
- Training conducted on a GPU-enabled environment
- Early stopping applied to prevent overfitting

---

## 📈 Evaluation Metrics

Model performance is evaluated using:
- **Precision**
- **Recall**
- **mAP@0.5**
- **mAP@0.5–0.95**

Evaluation is performed on both validation and test datasets to assess generalization.

---

## 🌐 Deployment

The trained model is deployed as a **Streamlit web application** with the following capabilities:

- Image upload inference
- Real-time webcam detection
- Adjustable confidence and IoU thresholds
- Class-based filtering
- Visual bounding box rendering

The application is designed for **inference only** and does not support training.

---

## 🖥️ Application Interface

Key components of the app:
- Sidebar controls for inference parameters
- Main view for image or webcam detection
- Tabular output of detected classes and confidence scores

---

## 📁 Project Structure
```
project-root/
├── app.py
├── classes.txt
├── requirements.txt
├── best.pt
├── data/
│   ├── raw/
│   ├── dataset_working/
│   └── dataset_final/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── report/
│   └── Report.pdf
├── Presentation.pptx
└── README.md
```

How to Run

1️⃣ Install dependencies
```
pip install -r requirements.txt
```

2️⃣ Run the application
```
streamlit run app.py
```

3️⃣ Open in browser
```
http://localhost:8501
```

## ⚠️ Limitations

- Performance depends on dataset quality and class balance

- Webcam mode requires local execution

- Nutrition estimation is not included in this version


## 🔮 Future Work

- Integrate food segmentation for portion estimation

- Integration of Calorie and Nutritional Estimation

- Improve class balance and data augmentation

- Creation of a Proprietar and High-Quality Dataset

- Expansion of Class Diversity for Comprehensive Coverage


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is intended for academic and research purposes.


## 👤 Authors

- [Author Nicholas Victorio]
- [Author Kenneth Owen]
- [Author Nixon Raine Vicsant]

Developed as part of a Deep Learning academic project.
