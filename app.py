import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# Set page configuration
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title("🔍 PCB Defect Detection System")
st.write("Upload a PCB image to detect defects")

# Function to download model from Google Drive
@st.cache_resource
def download_model():
    model_path = 'pcb_defect_model.h5'
    
    # Check if model already exists
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive... (This may take 30-60 seconds)")
        
        # Google Drive file ID extracted from your link
        file_id = '15NeEfT7106PH6RnolnhPdHWwHLMz49yC'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            # Download the model
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return None
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
with st.spinner('Loading AI model...'):
    model = download_model()

if model is None:
    st.error("⚠️ Model failed to load. Please contact support.")
else:
    st.success("✅ Model ready!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PCB image...", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Preprocess and predict
        with col2:
            st.subheader("🔮 Prediction Result")
            
            with st.spinner('Analyzing PCB...'):
                # Preprocess image
                img = image.resize((128, 128))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Determine result
                if prediction > 0.5:
                    result = "UNDEFECTIVE"
                    confidence = prediction * 100
                    color = "green"
                    icon = "✅"
                else:
                    result = "DEFECTIVE"
                    confidence = (1 - prediction) * 100
                    color = "red"
                    icon = "❌"
                
                # Display result
                st.markdown(f"### {icon} {result}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Progress bar for confidence
                st.progress(confidence / 100)
                
                # Additional info
                if result == "DEFECTIVE":
                    st.error("⚠️ Defect detected! This PCB needs inspection.")
                else:
                    st.success("✅ No defects found! PCB is good.")

    # Instructions
    st.markdown("---")
    st.markdown("### 📖 How to Use:")
    st.markdown("""
    1. Click **Browse files** above
    2. Select a PCB image from your computer
    3. Wait for the analysis (2-3 seconds)
    4. View the detection result
    """)

    # Footer
    st.markdown("---")
    st.markdown("**🔬 ML Model:** CNN with 3 Conv layers | **🎯 Accuracy:** ~85-90%")
    st.markdown("*Powered by TensorFlow & Streamlit*")
