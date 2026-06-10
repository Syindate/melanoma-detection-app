import streamlit as st
import numpy as np
import cv2
import torch
import importlib.util
from PIL import Image
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = importlib.util.spec_from_file_location("model", "model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    UNet = model_module.UNet

    model1 = UNet().to(device)
    model1.load_state_dict(torch.load("best_new_model.pth", map_location=device))
    model1.eval()

    model2 = UNet().to(device)
    model2.load_state_dict(torch.load("belowavgtuned_model.pth", map_location=device))
    model2.eval()

    return model1, model2, device

model1, model2, device = load_models()

# =========================
# PREDICT (DOĞRU)
# =========================
def predict(img_np):
    img_np = img_np.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2,0,1))

    x = torch.from_numpy(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(model1(x))
        p2 = torch.sigmoid(model2(x))

    p_ens = (p1 + p2) / 2
    return p_ens.squeeze().cpu().numpy()

# =========================
# CLASSIFIER
# =========================
def classify_3zone(conf):
    if conf > 0.465:
        return 1
    elif conf < 0.395:
        return 0
    else:
        return -1

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

    kmeans = KMeans(n_clusters=k, n_init=5).fit(hsv_pixels)
    return len(kmeans.cluster_centers_)


def diameter_score(mask):
    area = mask.sum()
    if area == 0:
        return 0

    d_px = np.sqrt(4 * area / np.pi)
    return d_px * 0.033

# =========================
# UI HEADER
# =========================
st.title("🧠 Melanoma Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file:

    # =========================
    # LOAD IMAGE
    # =========================
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 🔥 MODEL SIZE
    img_resized = cv2.resize(image, (256,256))

    # =========================
    # PREDICT
    # =========================
    p_ens = predict(img_resized)

    # 🔥 MASK FIX (EN KRİTİK)
    mask = (p_ens > 0.5).astype(np.uint8)

    # 🔥 MORPH CLEAN (noise fix)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # =========================
    # CONFIDENCE
    # =========================
    lesion_pixels = p_ens[mask == 1]

    if len(lesion_pixels) > 0:
        mean_conf = np.mean(lesion_pixels)
        std_conf = np.std(lesion_pixels)
    else:
        mean_conf = np.mean(p_ens)
        std_conf = 0

    area_ratio = mask.sum() / (256*256)
    confidence = (mean_conf - std_conf) * area_ratio

    # =========================
    # RISK
    # =========================
    risk = np.clip(confidence, 0, 1) * 100

    if risk > 65:
        risk_level = "HIGH"
    elif risk > 45:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # =========================
    # CLASSIFICATION
    # =========================
    pred = classify_3zone(confidence)

    if pred == 1:
        label = "MELANOMA"
    elif pred == 0:
        label = "BENIGN"
    else:
        label = "DOCTOR"

    # =========================
    # OVERLAY (FIX)
    # =========================
    overlay = img_resized.copy()
    overlay[mask == 1] = [255,0,0]

    # =========================
    # ABCD
    # =========================
    A = asymmetry_score(mask)
    B = border_score(mask)
    C = color_score(img_resized, mask)
    D = diameter_score(mask)

    # =========================
    # 🔥 UI GRID (PREMIUM)
    # =========================
    st.subheader("🔍 Visual Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_resized, caption="Original (256x256)", use_column_width=True)

    with col2:
        st.image(mask*255, caption="Segmentation Mask", use_column_width=True)

    with col3:
        st.image(overlay, caption="Overlay", use_column_width=True)

    # =========================
    # RESULTS
    # =========================
    st.subheader("📊 Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Result", label)

    with col2:
        st.metric("Confidence", f"{confidence:.4f}")

    with col3:
        st.metric("Risk", f"%{risk:.2f} ({risk_level})")

    # =========================
    # ABCD
    # =========================
    st.subheader("🧬 ABCD Analysis (Explainability)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Asymmetry", f"{A:.3f}")

    with col2:
        st.metric("Border", f"{B:.3f}")

    with col3:
        st.metric("Color", f"{C}")

    with col4:
        st.metric("Diameter (mm)", f"{D:.2f}")

    st.info("ABCD is for explainability only. Final decision uses model confidence.")