"""
app.py
------
SkinSense — Final production UI.
All predictions come exclusively from predictor.py → model.predict().
No changes to backend logic, preprocessing, or data flow.
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError
from predictor import load_model, load_label_encoder, predict

st.set_page_config(
    page_title="SkinSense",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #f8fafc; }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 760px;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.3px;
    }
    .hero p {
        font-size: 0.88rem;
        opacity: 0.85;
        margin: 0;
    }

    /* ── Upload wrapper ── */
    .upload-wrap {
        background: white;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem 0.5rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }

    /* ── Analyse button ── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 2px 10px rgba(79,70,229,0.3) !important;
        transition: opacity 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.87 !important;
    }

    /* ── Result cards ── */
    .res-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        height: 100%;
    }
    .res-card .lbl {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .res-card .val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.3;
    }
    .res-card .sub {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .res-card.accent {
        border-left: 4px solid #4f46e5;
        background: linear-gradient(135deg, #f5f3ff 0%, #faf5ff 100%);
    }
    .res-card.accent .val { color: #4338ca; }

    /* ── Probability bars ── */
    .bars-wrap {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-top: 0.8rem;
    }
    .bars-title {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }
    .bar-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.38rem;
    }
    .bar-name {
        width: 190px;
        font-size: 0.79rem;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex-shrink: 0;
    }
    .bar-name.top { color: #4338ca; font-weight: 600; }
    .bar-track {
        flex: 1;
        background: #f1f5f9;
        border-radius: 99px;
        height: 7px;
        overflow: hidden;
    }
    .bar-fill {
        height: 7px;
        border-radius: 99px;
        background: #e2e8f0;
    }
    .bar-fill.top {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
    }
    .bar-pct {
        width: 40px;
        text-align: right;
        font-size: 0.77rem;
        color: #94a3b8;
        flex-shrink: 0;
    }
    .bar-pct.top { color: #4f46e5; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Class labels ───────────────────────────────────────────────────────────────
CLASS_LABELS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

def get_readable_name(code: str) -> str:
    return CLASS_LABELS.get(code, code)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()

@st.cache_resource(show_spinner=False)
def get_encoder():
    return load_label_encoder()

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

label_encoder = get_encoder()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 SkinSense</h1>
    <p>Skin lesion classification &nbsp;·&nbsp; HAM10000 &nbsp;·&nbsp; 7 diagnostic classes &nbsp;·&nbsp; EfficientNet</p>
</div>
""", unsafe_allow_html=True)


# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG — resized to 224×224 internally"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image.load()
    except UnidentifiedImageError:
        st.error("Could not read the file as an image. Please upload a valid JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"Image error: {e}")
        st.stop()

    # Preview + meta + button
    col_img, col_action = st.columns([2, 3], gap="medium")

    with col_img:
        st.image(image, use_container_width=True, caption=f"{image.size[0]} × {image.size[1]} px")

    with col_action:
        st.markdown(f"""
        <div style="padding: 0.3rem 0 0.9rem 0;">
            <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:0.1rem;">File</div>
            <div style="font-size:0.92rem; font-weight:600; color:#0f172a;
                        margin-bottom:0.75rem;">{uploaded_file.name}</div>
            <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:0.1rem;">Dimensions</div>
            <div style="font-size:0.88rem; color:#475569;
                        margin-bottom:0.75rem;">{image.size[0]} × {image.size[1]} px · {image.mode}</div>
            <div style="font-size:0.72rem; color:#cbd5e1; margin-bottom:0.9rem;">
                Resized to 224 × 224 for inference
            </div>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = st.button("🔍  Analyse Image", type="primary", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if predict_btn:
        try:
            with st.spinner("Running inference..."):
                result = predict(image, model, label_encoder)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        top_code = result["label"]
        top_name = get_readable_name(top_code)
        confidence = result["confidence"]

        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

        # Result cards
        rc1, rc2 = st.columns(2, gap="medium")
        with rc1:
            st.markdown(f"""
            <div class="res-card accent">
                <div class="lbl">Predicted Class</div>
                <div class="val">{top_name}</div>
                <div class="sub">code: {top_code}</div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="res-card">
                <div class="lbl">Confidence</div>
                <div class="val">{confidence}%</div>
                <div class="sub">top-1 softmax probability</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        sorted_scores = sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        bars_html = '<div class="bars-wrap"><div class="bars-title">All Class Probabilities</div>'
        for code, score in sorted_scores:
            readable = get_readable_name(code)
            is_top = (code == top_code)
            bars_html += f"""
            <div class="bar-row">
                <div class="bar-name {'top' if is_top else ''}">
                    <span style="font-weight:600;">{code}</span> — {readable}
                </div>
                <div class="bar-track">
                    <div class="bar-fill {'top' if is_top else ''}" style="width:{score}%;"></div>
                </div>
                <div class="bar-pct {'top' if is_top else ''}">{score}%</div>
            </div>"""
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)
