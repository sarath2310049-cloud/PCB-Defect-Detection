import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title("🔍 PCB Defect Detection System")
st.write("Upload a PCB image or receive from Basler Camera")

# Function to download model from Google Drive
@st.cache_resource
def download_model():
    model_path = 'pcb_defect_model.h5'
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive... (This may take 30-60 seconds)")
        file_id = '15NeEfT7106PH6RnolnhPdHWwHLMz49yC'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ---- NEW: prediction function (reusable) ----
def run_prediction(image, model):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 PCB Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔮 Prediction Result")
        with st.spinner('Analyzing PCB...'):
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)[0][0]
            prediction = float(prediction)

            if prediction > 0.5:
                result = "UNDEFECTIVE"
                confidence = prediction * 100
                icon = "✅"
            else:
                result = "DEFECTIVE"
                confidence = (1 - prediction) * 100
                icon = "❌"

            st.markdown(f"### {icon} {result}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            st.progress(float(confidence / 100))

            if result == "DEFECTIVE":
                st.error("⚠️ Defect detected! This PCB needs inspection.")
            else:
                st.success("✅ No defects found! PCB is good.")

# Load model
with st.spinner('Loading AI model...'):
    model = download_model()

if model is None:
    st.error("⚠️ Model failed to load. Please contact support.")
else:
    st.success("✅ Model ready!")

    # ---- NEW: Check if image came from Basler camera via URL ----
    img_url = st.query_params.get("img_url", None)

    if img_url:
        st.info("📡 Image received from Basler Camera!")
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            run_prediction(image, model)
        except Exception as e:
            st.error(f"❌ Failed to load camera image: {e}")

    else:
        # Normal manual upload (your existing flow)
        uploaded_file = st.file_uploader(
            "Choose a PCB image...",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            run_prediction(image, model)

    # Instructions
    st.markdown("---")
    st.markdown("### 📖 How to Use:")
    st.markdown("""
    1. Click **Browse files** above for manual upload
    2. OR connect Basler Camera — image will load automatically
    3. Wait for the analysis (2-3 seconds)
    4. View the detection result
    """)

    # Footer
    st.markdown("---")
    st.markdown("**🔬 ML Model:** CNN with 3 Conv layers | **🎯 Accuracy:** ~85-90%")
    st.markdown("*Powered by TensorFlow & Streamlit*")
