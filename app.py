import io

import streamlit as st
from PIL import Image

from src.predict_sign import predict


def is_valid_image(data: bytes) -> bool:
    """Accept only real PNG or JPEG files (the extension can lie)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return img.format in ("PNG", "JPEG")
    except Exception:
        return False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Signify | Alphabet Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --purple: #7357D9;
    --purple-dark: #5137AC;
    --lavender: #F3F0FF;
    --ink: #211D35;
    --muted: #77738A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(145deg, #FBFAFF 0%, #F5F2FF 55%, #FDFBFF 100%);
    color: var(--ink);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: #F0ECFF;
    border-right: 1px solid #E4DDFB;
}

[data-testid="stSidebar"] h2 {
    color: #5137AC;
}

.hero {
    background: linear-gradient(120deg, #6548C9 0%, #8D75EA 58%, #B5A4FA 100%);
    padding: 36px 38px;
    border-radius: 26px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 32px rgba(92, 66, 181, 0.16);
}

.hero h1 {
    color: white !important;
    font-size: 2.65rem;
    line-height: 1.15;
    margin: 10px 0;
}

.hero p {
    color: #F3EEFF;
    font-size: 1.05rem;
    max-width: 650px;
    margin-bottom: 0;
}

.eyebrow {
    font-size: 0.76rem;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
    color: #E8DFFF;
}

.panel {
    background: rgba(255,255,255,0.92);
    border: 1px solid #E9E3FA;
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 6px 22px rgba(50, 35, 95, 0.045);
    min-height: 245px;
}

.panel-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #29223F;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: #817B95;
    font-size: 0.91rem;
    margin-bottom: 22px;
}

.upload-note {
    background: #F5F1FF;
    border: 1px dashed #B7A6F0;
    border-radius: 16px;
    padding: 17px;
    color: #5B4A8E;
    text-align: center;
    margin-top: 12px;
}

.result-box {
    background: linear-gradient(145deg, #F2EEFF, #FBF9FF);
    border: 1px solid #DCD1FF;
    border-radius: 20px;
    padding: 22px;
    text-align: center;
    margin-top: 16px;
}

.result-letter {
    font-family: 'Outfit', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1.15;
    color: #684BC9;
}

.result-caption {
    color: #77708E;
    font-size: 0.9rem;
    margin-top: 4px;
}

.tip {
    background: #FFFFFF;
    border: 1px solid #EAE5F8;
    border-radius: 14px;
    padding: 15px 17px;
    color: #625C76;
    font-size: 0.91rem;
}

.stButton > button {
    background: linear-gradient(100deg, #7357D9, #8D71E8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.72rem 1.1rem;
    font-weight: 700;
    min-height: 48px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #583CB8;
    color: white;
    border: none;
    box-shadow: 0 5px 15px rgba(93, 66, 190, 0.2);
}

div[data-testid="stFileUploader"] {
    background: #FCFBFF;
    border-radius: 15px;
}

hr {
    border-color: #E7E0F7;
}

.footer {
    text-align: center;
    color: #928AA8;
    font-size: 0.84rem;
    padding-top: 24px;
}

@media (max-width: 700px) {
    .hero {
        padding: 26px 22px;
    }
    .hero h1 {
        font-size: 2rem;
    }
    .panel {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🤟 Signify")
    st.caption("SIGN LANGUAGE • AI PROJECT")
    st.divider()

    st.markdown("### About the project")
    st.write(
        "An alphabet recognition prototype that uses "
        "computer vision (SIFT features) and an SVM classifier."
    )

    st.markdown("### How it works")
    st.markdown("""
    **01** · Upload a hand-sign image

    **02** · Extract SIFT features

    **03** · Predict with SVM

    **04** · View the predicted letter
    """)

    st.divider()
    st.markdown("### Tips for better inputs")
    st.markdown("""
    - Use a clear, well-lit image.
    - Keep the hand fully visible.
    - Avoid blurry images.
    - Use a plain background when possible.
    """)

    st.divider()
    st.caption("Model: SIFT + Bag of Words + SVM")
    st.caption("Interface: Streamlit")


# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION · MACHINE LEARNING</div>
    <h1>Every gesture tells<br> a story.</h1>
    <p>
        Explore sign language alphabet recognition.
        Upload a hand-sign image and let the trained model
        predict the corresponding letter.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- MAIN CONTENT ----------------
left, right = st.columns([1.05, 0.95], gap="large")

# Read and check the upload first, so both columns can use the result.
uploaded_file = None
image_bytes = None
file_ok = False

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📤 Upload your sign</div>
        <div class="panel-subtitle">
            Choose a clear image in PNG, JPG or JPEG format.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        file_ok = is_valid_image(image_bytes)
        if not file_ok:
            st.error("Please upload a real PNG or JPG image.")

    st.markdown("""
    <div class="upload-note">
        <b>Best results</b><br>
        Use a visible hand gesture with good lighting.
    </div>
    """, unsafe_allow_html=True)

    st.caption("Works best on a clear single-hand image.")

    predict_clicked = st.button(
        "✨  Predict alphabet",
        type="primary",
        width="stretch",
        disabled=not file_ok,
    )

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">🔎 Recognition result</div>
        <div class="panel-subtitle">
            Your predicted alphabet will appear here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Always show the original color image.
    if file_ok:
        st.image(
            image_bytes,
            caption="Selected hand-sign image",
            width="stretch",
        )

    if predict_clicked and file_ok:
        try:
            with st.spinner("Analysing the hand sign..."):
                result = predict(image_bytes)

            if result["status"] == "ok":
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-caption">PREDICTED ALPHABET</div>
                    <div class="result-letter">{result['label']}</div>
                    <div class="result-caption">
                        Confidence: {result['confidence']:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success("Prediction completed successfully.")
            else:
                # not_a_sign, low_confidence, no_features or invalid_file
                st.warning(result["message"])
                if result.get("best_guess"):
                    st.caption(
                        f"Closest match: {result['best_guess']} "
                        f"({result['confidence']:.0%}), "
                        "but the model is not sure enough."
                    )

        except FileNotFoundError as error:
            st.error(f"Model file is missing: {error}")
        except Exception as error:
            st.error(f"Unable to predict this image: {error}")

    elif uploaded_file is None:
        st.markdown("""
        <div class="result-box">
            <div style="font-size: 3rem;">✦</div>
            <div style="font-weight: 700; color: #684BC9;">
                Your result awaits
            </div>
            <div class="result-caption">
                Upload an image and select Predict alphabet.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- WEBCAM CAPTURE (PERSON 4) ----------------
st.markdown("---")
st.markdown("## Try with your webcam")
st.caption("Capture a clear, single-hand sign using your camera.")

photo = st.camera_input("Capture a hand sign")

if photo is not None:
    camera_bytes = photo.getvalue()

    if not is_valid_image(camera_bytes):
        st.error("Please capture a valid PNG or JPG image.")
    else:
        st.image(
            camera_bytes,
            caption="Captured hand-sign image",
            use_container_width=True,
        )

        if st.button("Predict captured sign", type="primary"):
            try:
                with st.spinner("Analysing the captured image..."):
                    camera_result = predict(camera_bytes)

                if camera_result["status"] == "ok":
                    st.success(
                        f"Predicted alphabet: {camera_result['label']}"
                    )
                    st.write(
                        f"Confidence: {camera_result['confidence']:.0%}"
                    )
                else:
                    st.warning(camera_result["message"])

            except FileNotFoundError as error:
                st.error(f"Model file is missing: {error}")
            except Exception as error:
                st.error(f"Unable to predict this image: {error}")



# ---------------- HOW IT WORKS ----------------
st.markdown("---")
st.markdown("## How the recognition works")
st.caption("A simple pipeline from image input to alphabet prediction.")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="tip">
        <b>📷 1. Image input</b><br><br>
        The app checks your upload is a real PNG or JPG image.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="tip">
        <b>🧩 2. Feature extraction</b><br><br>
        SIFT keypoints are grouped into visual words and counted in a histogram.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="tip">
        <b>🤖 3. SVM prediction</b><br><br>
        The classifier predicts a letter, or rejects images that are not hand signs.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    SIGNIFY · Sign Language Alphabet Recognition<br>
    Built with Python, OpenCV, SIFT, scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)