import streamlit as st
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

#page setup
st.set_page_config(
    page_title="Visual Guessing Machine",
    page_icon="✦",
    layout="centered"
)

#custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background:
        radial-gradient(circle at top left, rgba(140, 90, 255, 0.34), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 110, 210, 0.22), transparent 28%),
        radial-gradient(circle at bottom center, rgba(90, 220, 255, 0.16), transparent 34%),
        linear-gradient(180deg, #080012 0%, #130021 45%, #090014 100%);
    color: #f4ecff;
    font-family: 'Inter', sans-serif;
}

.title-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.9rem;
    margin-top: 0.2rem;
    margin-bottom: 1.05rem;
    flex-wrap: nowrap;
    white-space: nowrap;
}

.star-icon {
    width: 32px;
    height: 32px;
    flex: 0 0 auto;
    filter:
        drop-shadow(0 0 8px rgba(210, 170, 255, 0.95))
        drop-shadow(0 0 18px rgba(123, 231, 255, 0.45));
    opacity: 0.98;
}

.main-title {
    font-size: 3.7rem;
    font-weight: 600;
    text-align: center;
    color: #ffffff;
    font-family: 'Cormorant Garamond', serif;
    letter-spacing: 0.04em;
    line-height: 1;
    white-space: nowrap;
    text-shadow:
        0 0 10px rgba(255, 255, 255, 0.25),
        0 0 22px rgba(190, 145, 255, 0.78),
        0 0 38px rgba(123, 231, 255, 0.18);
}

.sub-title {
    text-align: center;
    font-size: 1.38rem;
    color: #eadbff;
    margin-bottom: 2.2rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

.info-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 25px rgba(120, 80, 255, 0.18);
}

.result-card {
    background: linear-gradient(135deg, rgba(120, 80, 255, 0.18), rgba(255, 80, 180, 0.12));
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
}

.small-label {
    color: #f3d7ff;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}

.top-prediction {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
}

.caption-line {
    color: #d6c6f8;
    font-size: 0.95rem;
}

.pred-row {
    background: rgba(255, 255, 255, 0.06);
    padding: 0.75rem 0.9rem;
    border-radius: 14px;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.pred-label {
    font-weight: 600;
    color: #ffffff;
}

.pred-score {
    color: #d8b8ff;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 0.5rem;
    border: 1px solid rgba(255,255,255,0.1);
}

img {
    border-radius: 18px;
}

@media (max-width: 700px) {
    .main-title {
        font-size: 2.75rem;
    }

    .star-icon {
        width: 24px;
        height: 24px;
    }

    .sub-title {
        font-size: 1.15rem;
        margin-bottom: 1.8rem;
    }

    .title-wrap {
        gap: 0.55rem;
        margin-bottom: 0.9rem;
    }
}
</style>
""", unsafe_allow_html=True)

#load the pretrained weights and model
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.eval()

#use the preprocessing that matches the pretrained weights
preprocess = weights.transforms()
categories = weights.meta["categories"]

#header
st.markdown("""
<div class="title-wrap">
    <svg class="star-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 8L59 39L92 50L59 61L50 92L41 61L8 50L41 39L50 8Z" fill="url(#grad1)"/>
        <defs>
            <linearGradient id="grad1" x1="0" y1="0" x2="100" y2="100">
                <stop offset="0%" stop-color="#ffffff"/>
                <stop offset="45%" stop-color="#e0c6ff"/>
                <stop offset="100%" stop-color="#7be7ff"/>
            </linearGradient>
        </defs>
    </svg>
    <div class="main-title">Visual Guessing Machine</div>
    <svg class="star-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 8L59 39L92 50L59 61L50 92L41 61L8 50L41 39L50 8Z" fill="url(#grad2)"/>
        <defs>
            <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="100">
                <stop offset="0%" stop-color="#ffffff"/>
                <stop offset="45%" stop-color="#e0c6ff"/>
                <stop offset="100%" stop-color="#7be7ff"/>
            </linearGradient>
        </defs>
    </svg>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">pokemon? plane? sandwich? let’s find out</div>',
    unsafe_allow_html=True
)

#instructions
st.markdown("""
<div class="info-card">
    <div class="small-label">instructions</div>
    <div>upload an image and this app will classify it with a pretrained torchvision model</div>
    <div style="margin-top:0.45rem;">supported image types: jpg, jpeg, png</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    #show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="uploaded image", use_container_width=True)

    #turn image into model input
    img_tensor = preprocess(image).unsqueeze(0)

    #make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    #get top 5 predictions
    top_probs, top_idxs = torch.topk(probs, 5)

    #show main result
    top_label = categories[top_idxs[0].item()]
    top_score = top_probs[0].item() * 100

    st.markdown(f"""
    <div class="result-card">
        <div class="small-label">model's best guess</div>
        <div class="top-prediction">{top_label}</div>
        <div class="caption-line">{top_score:.2f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    #show top predictions
    st.subheader("top predictions")
    for score, idx in zip(top_probs, top_idxs):
        label = categories[idx.item()]
        percent = score.item() * 100
        st.markdown(f"""
        <div class="pred-row">
            <span class="pred-label">{label}</span>
            <span class="pred-score"> — {percent:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)