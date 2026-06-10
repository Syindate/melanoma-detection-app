import streamlit as st
import numpy as np
import cv2
import torch
import importlib.util
from PIL import Image
from sklearn.cluster import KMeans

# =========================
# 🎨 STYLE
# =========================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: white;}

.card {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 15px;
    text-align: center;
}

img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Melanoma Detection System")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = importlib.util.spec_from_file_location("model", "./model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    UNet = model_module.UNet

    model1 = UNet().to(device)
    model1.load_state_dict(torch.load("./best_new_model.pth", map_location=device))
    model1.eval()

    model2 = UNet().to(device)
    model2.load_state_dict(torch.load("./belowavgtuned_model.pth", map_location=device))
    model2.eval()

    return model1, model2, device

model1, model2, device = load_models()

# =========================
# PREDICT (COLAB SAME)
# =========================
def predict(img_np):

    img_np = img_np.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2,0,1))

    x = torch.from_numpy(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = model1(x)
        p2 = model2(x)

    p = (p1 + p2) / 2

    return p.squeeze().cpu().numpy()

# =========================
# CLASSIFIER
# =========================
def classify(conf):
    if conf > 0.465:
        return "MELANOMA"
    elif conf < 0.395:
        return "BENIGN"
    else:
        return "DOCTOR"

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file:

    # 🔥 ORİJİNALİ KORU
    image = Image.open(uploaded_file).convert("RGB")

    # 🔥 MODEL INPUT (256)
    img_resized = image.resize((256,256))
    img_np = np.array(img_resized)

    # =========================
    # MODEL
    # =========================
    p = predict(img_np)

    # 🔥 DOĞRU MASK (NO POSTPROCESS)
    mask = (p > 0.5).astype(np.uint8)

    # =========================
    # OVERLAY
    # =========================
    overlay = img_np.copy()
    overlay[mask == 1] = [255,0,0]

    # =========================
    # 🔥 DISPLAY SIZE FIX
    # =========================
    display_size = (350,350)

    img_display = cv2.resize(img_np, display_size)
    mask_display = cv2.resize(mask*255, display_size, interpolation=cv2.INTER_NEAREST)
    overlay_display = cv2.resize(overlay, display_size)

    # =========================
    # CONFIDENCE
    # =========================
    lesion_pixels = p[mask == 1]

    if len(lesion_pixels) > 0:
        mean_conf = np.mean(lesion_pixels)
        std_conf = np.std(lesion_pixels)
    else:
        mean_conf = np.mean(p)
        std_conf = 0

    area_ratio = mask.sum() / (256*256)
    confidence = (mean_conf - std_conf) * area_ratio
    risk = np.clip(confidence, 0, 1) * 100

    result = classify(confidence)

    # =========================
    # UI
    # =========================
    st.subheader("Visual Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_display, caption="Original", use_container_width=True)

    with col2:
        st.image(mask_display, caption="Segmentation Mask", use_container_width=True)

    with col3:
        st.image(overlay_display, caption="Overlay", use_container_width=True)

    # =========================
    # METRICS
    # =========================
    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)

    c1.metric("Result", result)
    c2.metric("Confidence", f"{confidence:.4f}")
    c3.metric("Risk", f"%{risk:.2f}")