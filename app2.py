import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load your trained YOLO model
model = YOLO(r"C:\Users\prath\runs\detect\runway_poc4\weights\best.pt")  # <-- use raw string

st.title("Airport Runway Crack Detection 🚀")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run YOLO prediction
    results = model.predict(source=img_array, conf=0.25, save=False)

    # Get annotated image
    annotated_img = results[0].plot()  # returns image with boxes

    # Convert to PIL and show in Streamlit
    annotated_pil = Image.fromarray(annotated_img)
    st.image(annotated_pil, caption="Detection Result", use_column_width=True)
