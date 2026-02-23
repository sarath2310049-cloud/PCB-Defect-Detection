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
        st.info("📥 Downloading updated model from Google Drive... (This may take 30-60 seconds)")
        
        # NEW Google Drive file ID from updated model
        file_id = '1goyY8DHvX3dXkvW3q4bPV4lzmEW9M8iY'
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
        try:
            # Load and convert image to RGB
            image = Image.open(uploaded_file)
            
            # CRITICAL: Convert to RGB (handles grayscale, RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display the uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Uploaded Image")
                st.image(image, use_container_width=True)
            
            # Preprocess and predict
            with col2:
                st.subheader("🔮 Prediction Result")
                
                with st.spinner('Analyzing PCB...'):
                    # Resize to model input size
                    img = image.resize((128, 128))
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    # Verify shape is correct (128, 128, 3)
                    if img_array.shape != (128, 128, 3):
                        st.error(f"❌ Invalid image shape: {img_array.shape}. Expected (128, 128, 3)")
                    else:
                        # Add batch dimension (1, 128, 128, 3)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        # Convert to float (fix for progress bar)
                        prediction = float(prediction)
                        
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
                        
                        # Progress bar for confidence (convert to float)
                        st.progress(float(confidence / 100))
                        
                        # Additional info
                        if result == "DEFECTIVE":
                            st.error("⚠️ Defect detected! This PCB needs inspection.")
                        else:
                            st.success("✅ No defects found! PCB is good.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check if the file is corrupted.")

    # Instructions
    st.markdown("---")
    st.markdown("### 📖 How to Use:")
    st.markdown("""
    1. Click **Browse files** above
    2. Select a PCB image from your computer (JPG, JPEG, or PNG)
    3. Wait for the analysis (2-3 seconds)
    4. View the detection result with confidence score
    """)

    # Footer
    st.markdown("---")
    st.markdown("**🔬 ML Model:** CNN with 3 Conv layers | **🎯 Accuracy:** ~85-90% | **Version:** 2.0")
    st.markdown("*Powered by TensorFlow & Streamlit*")
