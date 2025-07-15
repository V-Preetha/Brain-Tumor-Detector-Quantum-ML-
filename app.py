import streamlit as st
from PIL import Image
import numpy as np
import torch
from qnn_model import HybridQNN

# === Page Config ===
st.set_page_config(page_title="Quantum Brain Tumor Detector", page_icon="🧠")

# === Constants ===
IMG_SIZE = (4,4)  # Must match training image size
MODEL_PATH = "trained_qnn_model.pth"

# === Load model (cached) ===
@st.cache_resource
def load_model():
    model = HybridQNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    tensor = torch.tensor(image.flatten()).unsqueeze(0)  # Add batch dimension
    return tensor

# === Prediction ===
def predict(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor)
        confidence = output.item()
        is_tumor = confidence > 0.5
    return is_tumor, confidence

# === UI ===
st.title("🧠 Brain Tumor Detector (Quantum ML)")
st.markdown("Upload a brain scan image to detect the presence of a tumor using a Quantum Neural Network (QNN).")

uploaded_file = st.file_uploader("📤 Upload a Brain Scan", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    st.markdown("🔍 **Analyzing with Quantum Neural Network...**")
    pred, conf = predict(img)

    if pred:
        st.markdown(f"<h3 style='color:red;'>🧠 Tumor Detected</h3>", unsafe_allow_html=True)
        st.progress(conf)
        st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
    else:
        st.markdown(f"<h3 style='color:green;'>✅ No Tumor Detected</h3>", unsafe_allow_html=True)
        st.progress(1 - conf)
        st.markdown(f"**Confidence:** `{(1 - conf) * 100:.2f}%`")

st.markdown("---")
st.caption("⚛️ Powered by Quantum Neural Networks • Developed during IITM Quantum Internship")

