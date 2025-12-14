# 🍽️ Food Detection App (YOLO + Streamlit)

A web-based food detection application built using **YOLO (Ultralytics)** and **Streamlit**.  
The app supports **image upload** and **real-time webcam inference** with configurable detection parameters.

---

## 🚀 Features

- Food object detection using YOLO11s
- Image upload inference
- Real-time webcam detection
- Confidence & IoU threshold control
- Class-based filtering
- Lightweight deployment (inference-only)

---

## 🧰 Requirements

### Python Version
- **Python 3.10 (recommended)**
- Python >=3.10 and <3.12

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt


## 📁 Required Files & Structure

Ensure the following files exist before running the app:

project-root/
├── app.py
├── classes.txt
├── requirements.txt
├── best.pt

classes.txt format
0 tempe_goreng
1 tahu_goreng
2 nasi_putih
...

## ▶️ How to Run the App (Local)

- Activate virtual environment

source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows


- Run Streamlit app

streamlit run app.py


- Open browser: http://localhost:8501


## 🖼️ How to Use
- Image Mode
- Select Mode → Image
- Upload image (.jpg, .png, .webp)
- Adjust confidence / IoU if needed
- Click Run detection
- Detection results and table will be displayed

- Webcam Mode
- Select Mode → Webcam
- Click Start webcam
- Click Stop webcam to end

## 🎛️ Parameters
- Confidence:	Minimum confidence threshold
- IoU (NMS): 	Non-Max Suppression threshold
- Max detections:	Maximum objects per image
- Class filter:	Detect all or selected classes

## 🔌 Endpoints (Streamlit)

This application uses Streamlit UI only
(no REST API endpoints exposed).

- Web UI:	http://localhost:8501
- Image inference:	UI-based
- Webcam inference:	UI-based

## 🌐 Deployment Options
- Local Deployment

Recommended for evaluation and demo: streamlit run app.py