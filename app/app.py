import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Food Detection - YOLO", layout="wide")


WEIGHTS_PATH = "best.pt"
CLASSES_FILE = "classes.txt"


@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
    return YOLO(WEIGHTS_PATH)

@st.cache_data
def load_class_names(path: str):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id, name = line.split(maxsplit=1)
            mapping[int(cls_id)] = name
    return mapping

def bgr_from_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def render_results_bgr(results, class_names):
    results[0].names = class_names
    return results[0].plot()

def run_inference_on_bgr(
    model, img_bgr: np.ndarray, conf: float, iou: float, max_det: int, classes=None
):
    return model.predict(
        source=img_bgr, conf=conf, iou=iou, max_det=max_det, classes=classes, verbose=False
    )


try:
    model = load_model()
    class_names = load_class_names(CLASSES_FILE)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()


st.markdown("## 🍽️ Food Detection (YOLO)")

with st.sidebar:
    st.markdown("### Model : YOLO11s")
    st.divider()
    st.header("Inference")
    conf = st.slider("Confidence", 0.01, 1.0, 0.25, 0.01)
    iou = st.slider("IoU (NMS)", 0.01, 1.0, 0.60, 0.01)
    max_det = st.slider("Max detections per image", 1, 300, 100, 1)
    st.header("Classes")
    class_filter_mode = st.radio(
        "Class filtering", ["All classes", "Only selected classes"], index=0
    )
    mode = st.radio("Mode", ["Image", "Webcam"], index=0)


selected_classes = None
if class_filter_mode == "Only selected classes":
    options = [f"{k}: {v}" for k, v in class_names.items()]
    chosen = st.multiselect("Select classes to detect", options)
    if chosen:
        selected_classes = [int(x.split(":")[0]) for x in chosen]
    else:
        selected_classes = None


if mode == "Image":
    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        run_btn = st.button("Run detection", type="primary", disabled=(uploaded is None))

    if uploaded and run_btn:
        img_bgr = bgr_from_bytes(uploaded.getvalue())
        results = run_inference_on_bgr(
            model=model,
            img_bgr=img_bgr,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=selected_classes,
        )
        annotated_bgr = render_results_bgr(results, class_names)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.image(annotated_rgb, caption="Detections", use_container_width=True)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            rows = []
            for b in boxes:
                cls_id = int(b.cls.item())
                rows.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_names.get(cls_id, "unknown"),
                        "confidence": round(float(b.conf.item()), 4),
                    }
                )
            st.subheader("Detections")
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No detections found.")


else:
    start = st.button("Start webcam", type="primary")
    stop = st.button("Stop webcam")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if start:
        st.session_state.webcam_running = True
    if stop:
        st.session_state.webcam_running = False

    frame_box = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break
            results = run_inference_on_bgr(
                model=model,
                img_bgr=frame,
                conf=conf,
                iou=iou,
                max_det=max_det,
                classes=selected_classes,
            )
            annotated = render_results_bgr(results, class_names)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_box.image(annotated_rgb, channels="RGB")
        cap.release()
        st.info("Webcam stopped.")