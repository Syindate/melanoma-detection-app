import streamlit as st
import numpy as np
import cv2
import torch
import importlib.util
from PIL import Image
from sklearn.cluster import KMeans

# =========================
# 🎨 STYLE (PREMIUM LOOK)
# =========================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: white;}
    .card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Melanoma Detection System")
st.caption("Upload a skin lesion image to analyze melanoma risk.")

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
# PREDICT
# =========================
def predict_fast(img_np):

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
def classify_3zone(conf):
    if conf > 0.465:
        return "MELANOMA"
    elif conf < 0.395:
        return "BENIGN"
    else:
        return "DOCTOR"

# =========================
# ABCD
# =========================
def asymmetry_score(mask):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return 0

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = mask[y0:y1, x0:x1]

    flip_h = np.fliplr(cropped)
    flip_v = np.flipud(cropped)

    diff_h = np.logical_xor(cropped, flip_h).sum()
    diff_v = np.logical_xor(cropped, flip_v).sum()

    total = cropped.sum() + 1e-8
    return (diff_h + diff_v) / (2 * total)

def border_score(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    P = cv2.arcLength(cnt, True)
    A = cv2.contourArea(cnt) + 1e-8

    return (P**2) / (4 * np.pi * A)

def color_score(image, mask, k=4):
    lesion_pixels = image[mask == 1]
    if len(lesion_pixels) < k:
        return 0

    hsv_pixels = cv2.cvtColor(
        lesion_pixels.reshape(-1,1,3).astype(np.uint8),
        cv2.COLOR_RGB2HSV
    ).reshape(-1,3)

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(hsv_pixels)
    return len(kmeans.cluster_centers_)

def diameter_score(mask):
    area = mask.sum()
    if area == 0:
        return 0

    d_px = np.sqrt(4 * area / np.pi)
    return d_px * 0.033

# =========================
# UI INPUT
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (256,256))

    # =========================
    # MODEL
    # =========================
    p = predict_fast(img_resized)

    mask = (p > 0.5).astype(np.uint8)
    mask = cv2.medianBlur(mask*255, 5)
    mask = (mask > 127).astype(np.uint8)

    overlay = img_resized.copy()
    overlay[mask == 1] = [255,0,0]

    # =========================
    # METRICS
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
    risk_percent = np.clip(confidence, 0, 1) * 100

    result = classify_3zone(confidence)

    A = asymmetry_score(mask)
    B = border_score(mask)
    C = color_score(img_resized, mask)
    D = diameter_score(mask)

    # =========================
    # 🖼️ IMAGE GRID (YAN YANA)
    # =========================
    st.subheader("🔍 Visual Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Original")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(mask*255, caption="Segmentation Mask")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(overlay, caption="Overlay")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # 📊 RESULTS GRID
    # =========================
    st.subheader("📊 Prediction")

    c1, c2, c3 = st.columns(3)

    c1.metric("Result", result)
    c2.metric("Confidence", f"{confidence:.4f}")
    c3.metric("Risk", f"%{risk_percent:.2f}")

    # =========================
    # 🧬 ABCD
    # =========================
    st.subheader("🧬 ABCD Analysis")

    a1, a2, a3, a4 = st.columns(4)

    a1.metric("Asymmetry", f"{A:.3f}")
    a2.metric("Border", f"{B:.3f}")
    a3.metric("Color", C)
    a4.metric("Diameter (mm)", f"{D:.2f}")

    st.info("ABCD is for explainability only. Final decision uses model confidence.")