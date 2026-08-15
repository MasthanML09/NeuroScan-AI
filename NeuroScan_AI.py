import warnings, os, logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import json
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import time

st.set_page_config(
    page_title="NeuroScan AI", page_icon="🧠",
    layout="wide", initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important; color: #e8eaf0; font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(56,189,248,0.08) 0%, transparent 60%), #050810 !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { max-width: 1100px !important; padding: 3rem 2rem !important; }
.hero { text-align: center; padding: 2.5rem 0 3rem; }
.hero-badge {
    display: inline-block; background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25); color: #38bdf8;
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.18em;
    padding: 0.35rem 1rem; border-radius: 100px; margin-bottom: 1.4rem; text-transform: uppercase;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2.6rem, 6vw, 4.2rem);
    font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
    background: linear-gradient(135deg, #f0f4ff 30%, #38bdf8 70%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1rem;
}
.hero-sub { font-size: 1rem; color: rgba(200,210,230,0.55); font-weight: 300; max-width: 480px; margin: 0 auto; line-height: 1.7; }
.result-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 2rem 2.4rem; margin-top: 1rem; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
}
.tumor-badge { display: inline-block; font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; padding: 0.5rem 0; margin-bottom: 0.4rem; }
.tumor-positive { color: #f87171; }
.tumor-negative { color: #4ade80; }
.result-meta { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: rgba(200,210,230,0.4); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.confidence-number { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 600; color: #e8eaf0; }
.prob-row { display: flex; align-items: center; gap: 12px; margin-bottom: 0.75rem; }
.prob-label { font-family: 'DM Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: rgba(200,210,230,0.6); width: 95px; flex-shrink: 0; }
.prob-track { flex: 1; height: 6px; background: rgba(255,255,255,0.06); border-radius: 100px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 100px; }
.prob-val { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: rgba(200,210,230,0.5); width: 46px; text-align: right; flex-shrink: 0; }
.section-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase; color: rgba(200,210,230,0.3); margin-bottom: 1rem; margin-top: 2rem; }
.divider { height: 1px; background: rgba(255,255,255,0.05); margin: 1.5rem 0; }
.disclaimer { background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.15); border-radius: 12px; padding: 1rem 1.4rem; margin-top: 2rem; font-size: 0.8rem; color: rgba(251,191,36,0.7); line-height: 1.6; }
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem; }
.chip { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 100px; padding: 0.3rem 0.85rem; font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em; color: rgba(200,210,230,0.45); }
[data-testid="column"] { padding: 0 0.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE     = 128
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "brain_tumor_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

# Fallback labels (sorted = same order used during training)
FALLBACK_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_COLORS = {
    "glioma": "#f87171", "meningioma": "#fb923c",
    "pituitary": "#a78bfa", "notumor": "#4ade80",
}
CLASS_INFO = {
    "glioma":     "Gliomas arise from glial cells in the brain or spine. They are the most common primary brain tumors.",
    "meningioma": "Meningiomas form on the membranes surrounding the brain and spinal cord. Usually slow-growing.",
    "pituitary":  "Pituitary tumors develop in the pituitary gland at the brain's base. Often treatable with surgery.",
    "notumor":    "No tumor detected in this MRI scan. The image appears normal based on the model's analysis.",
}


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    """
    Model is Functional API (verified) — load_model() works directly.
    No weight-loading tricks needed.
    """
    model = load_model(MODEL_PATH)

    # Use label_map.json if present, else sorted fallback
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        unique_labels = [label_map[str(i)] for i in range(len(label_map))]
    else:
        unique_labels = FALLBACK_LABELS

    # VGG16 base for Grad-CAM
    base = model.get_layer('vgg16')

    return model, unique_labels, base


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(pil_image):
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────


def make_gradcam(img_batch, model, base, last_conv="block4_conv3"):
    try:
        last_conv_layer = base.get_layer(last_conv)

        # Stage 1: VGG's own input -> chosen conv layer's output
        last_conv_layer_model = Model(base.input, last_conv_layer.output)

        # Stage 2: replay everything after this layer (rest of VGG16 +
        # the outer model's classification head) on a fresh input
        classifier_input = tf.keras.Input(
            shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        vgg_layer_names = [l.name for l in base.layers]
        idx = vgg_layer_names.index(last_conv)
        for layer in base.layers[idx + 1:]:
            x = layer(x)

        base_index = model.layers.index(base)
        for layer in model.layers[base_index + 1:]:
            x = layer(x)

        classifier_model = Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_out = last_conv_layer_model(img_batch, training=False)
            tape.watch(conv_out)
            preds = classifier_model(conv_out, training=False)
            pred_idx = tf.argmax(preds[0])
            score = preds[:, pred_idx]

        grads = tape.gradient(score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None



def overlay_gradcam(pil_image, heatmap, alpha=0.45):
    raw     = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    h_img   = np.array(Image.fromarray(np.uint8(heatmap * 255)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
    colored = np.uint8(cm.jet(h_img / 255.0)[:, :, :3] * 255)
    return np.uint8((1 - alpha) * raw + alpha * colored)


# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(pil_image, model, unique_labels, base):
    batch   = preprocess(pil_image)
    probs   = model.predict(batch, verbose=0)[0]
    idx     = int(np.argmax(probs))
    label   = unique_labels[idx]
    conf    = float(probs[idx])
    heatmap = make_gradcam(batch, model, base)
    return label, conf, probs, heatmap


# ─── Prob bar ─────────────────────────────────────────────────────────────────
def prob_bar(label, prob, color, is_top):
    pct   = prob * 100
    alpha = "ff" if is_top else "55"
    return f"""
    <div class="prob-row">
        <span class="prob-label">{label}</span>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%;background:{color}{alpha};"></div>
        </div>
        <span class="prob-val">{pct:.1f}%</span>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">Deep Learning · VGG16 · Grad-CAM</div>
    <div class="hero-title">NeuroScan AI</div>
    <div class="hero-sub">Upload a brain MRI and get an instant tumor analysis —
    type, confidence, and visual explanation.</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model…"):
    try:
        model, unique_labels, base = load_resources()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error(f"**Model not found.** Expected at: `{MODEL_PATH}`\n\nPlace `brain_tumor_model.keras` next to `app.py`.")
    except Exception as e:
        model_loaded = False
        st.error(f"**Could not load model:** {e}")

if model_loaded:
    st.markdown('<div class="section-label">Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")

        with st.spinner("Analysing scan…"):
            time.sleep(0.3)
            label, conf, probs, heatmap = predict(pil_image, model, unique_labels, base)

        is_tumor  = label != "notumor"
        desc      = CLASS_INFO.get(label, "")
        disp_name = "No Tumor" if label == "notumor" else label.capitalize()
        badge_cls = "tumor-positive" if is_tumor else "tumor-negative"
        prefix    = "⚠ " if is_tumor else "✓ "

        col_img, col_res = st.columns([1, 1.15], gap="large")

        with col_img:
            st.markdown('<div class="section-label">Scan Preview</div>', unsafe_allow_html=True)
            if heatmap is not None:
                tab_orig, tab_cam = st.tabs(["Original", "Grad-CAM Overlay"])
                with tab_orig:
                    st.image(pil_image, use_container_width=True)
                with tab_cam:
                    st.image(overlay_gradcam(pil_image, heatmap),
                             use_container_width=True,
                             caption="Highlighted regions drove the prediction")
            else:
                st.image(pil_image, use_container_width=True)

            st.markdown(f"""
            <div class="chip-row">
                <span class="chip">VGG16 backbone</span>
                <span class="chip">128 × 128 input</span>
                <span class="chip">{uploaded.name}</span>
            </div>""", unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

            bars_html = "".join(
                prob_bar(lbl, float(probs[i]), CLASS_COLORS.get(lbl, "#888"), lbl == label)
                for i, lbl in enumerate(unique_labels)
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-meta">Prediction</div>
                <div class="tumor-badge {badge_cls}">{prefix}{disp_name}</div>
                <div class="divider"></div>
                <div class="result-meta">Confidence Score</div>
                <div class="confidence-number">{conf*100:.2f}%</div>
                <div class="divider"></div>
                <div class="result-meta">Probability Distribution</div>
                {bars_html}
                <div class="divider"></div>
                <div class="result-meta">About This Finding</div>
                <div style="font-size:0.85rem;color:rgba(200,210,230,0.6);line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠ <strong>Medical Disclaimer:</strong> This tool is for educational and research
            purposes only. It is <strong>not a substitute for professional medical diagnosis</strong>.
            Always consult a qualified neurologist or radiologist for clinical decisions.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:rgba(200,210,230,0.2);">
            <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;letter-spacing:0.12em;">
                AWAITING SCAN UPLOAD</div>
        </div>
        """, unsafe_allow_html=True)
