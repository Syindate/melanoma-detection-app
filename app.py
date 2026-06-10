import streamlit as st
import numpy as np
import cv2
import torch
import importlib.util
from PIL import Image
from sklearn.cluster import KMeans

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():

    PROJECT_PATH = "."

    spec = importlib.util.spec_from_file_location("model", f"{PROJECT_PATH}/model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    UNet = model_module.UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = UNet().to(device)
    model1.load_state_dict(torch.load(f"{PROJECT_PATH}/best_new_model.pth", map_location=device))
    model1.eval()

    model2 = UNet().to(device)
    model2.load_state_dict(torch.load(f"{PROJECT_PATH}/belowavgtuned_model.pth", map_location=device))
    model2.eval()

    return model1, model2, device


model1, model2, device = load_models()

# =========================
# PREDICT (FIXED)
# =========================
def predict(img_np):

    img = img_np.astype(np.float32) / 255.0   # 🔥 CRITICAL FIX
    img = np.transpose(img, (2,0,1))

    x = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(model1(x))
        p2 = torch.sigmoid(model2(x))

    return ((p1 + p2) / 2).squeeze().cpu().numpy()

# =========================
# CLASSIFIER (UNCHANGED)
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
def asymmetry(mask):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return 0

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    crop = mask[y0:y1, x0:x1]

    flip_h = np.fliplr(crop)
    flip_v = np.flipud(crop)

    diff = np.logical_xor(crop, flip_h).sum() + np.logical_xor(crop, flip_v).sum()

    return diff / (2 * (crop.sum() + 1e-8))


def border(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    P = cv2.arcLength(cnt, True)
    A = cv2.contourArea(cnt) + 1e-8

    return (P**2) / (4 * np.pi * A)


def diameter(mask):
    area = mask.sum()
    if area == 0:
        return 0

    return np.sqrt(4 * area / np.pi) * 0.033


# =========================
# UI
# =========================
st.title("🧠 Melanoma Detection System")
st.write("Upload a skin lesion image to analyze melanoma risk.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded:

    # =========================
    # ✅ FIXED IMAGE LOAD (CV2 STYLE)
    # =========================
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))

    st.image(img, caption="Original Image", width=400)

    # =========================
    # PREDICT
    # =========================
    p = predict(img)

    # 🔥 threshold (same as training)
    mask = (p > 0.5).astype(np.uint8)

    # =========================
    # CONFIDENCE (UNCHANGED)
    # =========================
    lesion_pixels = p[mask == 1]

    if len(lesion_pixels) > 0:
        mean_conf = np.mean(lesion_pixels)
        std_conf = np.std(lesion_pixels)
    else:
        mean_conf = np.mean(p)
        std_conf = 0

    area_ratio = mask.sum() / (256*256)
    confidence_final = (mean_conf - std_conf) * area_ratio

    # =========================
    # RISK
    # =========================
    risk_percent = np.clip(confidence_final, 0, 1) * 100

    if risk_percent > 65:
        risk_level = "HIGH"
    elif risk_percent > 45:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # =========================
    # DECISION
    # =========================
    decision = classify_3zone(confidence_final)

    # =========================
    # ABCD
    # =========================
    A = asymmetry(mask)
    B = border(mask)
    D = diameter(mask)

    # =========================
    # OVERLAY
    # =========================
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]

    # =========================
    # DEBUG (istersen sil)
    # =========================
    st.write("DEBUG → min:", float(p.min()))
    st.write("DEBUG → max:", float(p.max()))

    # =========================
    # VISUALS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.image(mask * 255, caption="Segmentation Mask", width=300)

    with col2:
        st.image(overlay, caption="Overlay", width=300)

    # =========================
    # RESULTS
    # =========================
    st.markdown("## 📊 Prediction")

    st.write(f"**Result:** {decision}")
    st.write(f"**Confidence:** {confidence_final:.4f}")
    st.write(f"**Risk Score:** %{risk_percent:.2f} ({risk_level})")

    # =========================
    # ABCD
    # =========================
    st.markdown("## 🧬 ABCD Analysis (Explainability)")

    st.write(f"A (Asymmetry): {A:.4f}")
    st.write(f"B (Border): {B:.4f}")
    st.write(f"D (Diameter): {D:.2f} mm")

    st.info("ABCD features are used for interpretability only. Final decision is based on model confidence.")