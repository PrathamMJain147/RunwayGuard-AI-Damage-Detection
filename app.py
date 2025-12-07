import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import os
from collections import Counter

# Title
st.title("🛣️ Road Damage Detection")
st.markdown("Upload a road image to detect damages using YOLOv8 model")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = r'C:\Users\prath\Downloads\train\train\India\runs\detect\train\weights\best.pt'
    return YOLO(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Run detection
    with st.spinner("Detecting..."):
        results = model(image)
        annotated_img = results[0].plot()

    # Show result image
    st.image(annotated_img, caption="Detected Image", use_container_width=True)

    # Extract detections
    detections = results[0].boxes
    class_ids = detections.cls.cpu().numpy().astype(int)
    class_names = model.names
    labels = [class_names[i] for i in class_ids]
    counts = Counter(labels)

    # Show detection counts
    if counts:
        st.subheader("🧾 Detected Damages")
        for label, count in counts.items():
            st.write(f"**{label}**: {count}")
    else:
        st.write("✅ No damage detected.")

    # Show meaning of each class
    st.markdown("""
    ---
    ### 📘 Class Label Legend:
    - **D00**: Longitudinal Crack  
    - **D10**: Transverse Crack  
    - **D20**: Alligator Crack  
    - **D40**: Pothole  
    """)
