import streamlit as st
import numpy as np
import cv2
import torch
import importlib.util
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
PROJECT_PATH = "."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    spec = importlib.util.spec_from_file_location("model", f"{PROJECT_PATH}/model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    UNet = model_module.UNet

    model1 = UNet().to(device)
    model1.load_state_dict(torch.load("best_new_model.pth", map_location=device))
    model1.eval()

    model2 = UNet().to(device)
    model2.load_state_dict(torch.load("belowavgtuned_model.pth", map_location=device))
    model2.eval()

    return model1, model2

model1, model2 = load_models()

# =========================
# PREDICT (DOĞRU PIPELINE)
# =========================
def predict(img_np):
    img_np = img_np.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2,0,1))
    x = torch.from_numpy(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(model1(x))
        p2 = torch.sigmoid(model2(x))

    return ((p1 + p2) / 2).squeeze().cpu().numpy()

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
# ABCD (EXPLAINABILITY)
# =========================
def asymmetry_score(mask):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return 0

    y0,x0 = coords.min(axis=0)
    y1,x1 = coords.max(axis=0)
    cropped = mask[y0:y1, x0:x1]

    diff = np.logical_xor(cropped, np.fliplr(cropped)).sum()
    return diff / (cropped.sum() + 1e-8)

def border_score(mask):
    contours,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    P = cv2.arcLength(cnt, True)
    A = cv2.contourArea(cnt) + 1e-8

    return (P**2) / (4*np.pi*A)

def color_score(image, mask):
    pixels = image[mask == 1]
    if len(pixels) < 10:
        return 0

    hsv = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_RGB2HSV).reshape(-1,3)
    return len(KMeans(n_clusters=4, n_init=5).fit(hsv).cluster_centers_)

def diameter_score(mask):
    area = mask.sum()
    if area == 0:
        return 0
    return np.sqrt(4*area/np.pi)*0.033

# =========================
# UI
# =========================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

st.title("🧠 Melanoma Detection System")
st.markdown("Upload a skin lesion image to analyze melanoma risk.")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (256,256))

    st.image(img, caption="Original Image", use_column_width=True)

    # =========================
    # MODEL
    # =========================
    p_ens = predict(img_resized)
    mask = (p_ens > 0.5).astype(np.uint8)

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

    risk = np.clip(confidence, 0, 1) * 100
    prediction = classify_3zone(confidence)

    # =========================
    # ABCD
    # =========================
    A = asymmetry_score(mask)
    B = border_score(mask)
    C = color_score(img_resized, mask)
    D = diameter_score(mask)

    # =========================
    # VISUALS
    # =========================
    overlay = img_resized.copy()
    overlay[mask == 1] = [255, 0, 0]

    col1, col2 = st.columns(2)

    with col1:
        st.image(mask*255, caption="Segmentation Mask")

    with col2:
        st.image(overlay, caption="Overlay")

    # =========================
    # RESULTS
    # =========================
    st.subheader("📊 Prediction")

    st.write(f"**Result:** {prediction}")
    st.write(f"**Confidence:** {confidence:.4f}")
    st.write(f"**Risk Score:** %{risk:.2f}")

    # =========================
    # ABCD
    # =========================
    st.subheader("🧬 ABCD Analysis (Explainability)")

    st.write(f"A (Asymmetry): {A:.3f}")
    st.write(f"B (Border): {B:.3f}")
    st.write(f"C (Color): {C}")
    st.write(f"D (Diameter): {D:.2f} mm")

    # =========================
    # NOTE
    # =========================
    st.info("⚠️ ABCD features are used for interpretability only. Final decision is based on model confidence.")