# 🔍 PCB Defect Detection System

A machine learning-powered web application for detecting defects in Printed Circuit Boards (PCBs).

## 🎯 Features

- Upload PCB images through web interface
- Real-time defect detection using CNN model
- Confidence score for predictions
- User-friendly interface
- Cloud-deployed for easy access

## 🧠 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Layers:** 3 Convolutional layers with MaxPooling
- **Input Size:** 128x128 RGB images
- **Output:** Binary classification (Defective / Undefective)
- **Accuracy:** ~85-90% on test data

## 🚀 Live Demo

[Add your Streamlit app link here after deployment]

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── pcb_defect_model.h5    # Trained ML model
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 💻 Local Setup (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Technologies Used

- **TensorFlow/Keras:** Deep learning framework
- **Streamlit:** Web application framework
- **Python:** Programming language
- **PIL:** Image processing

## 📊 Model Training

The model was trained on a PCB defect dataset with:
- Data augmentation (rotation, flipping, zoom)
- Early stopping to prevent overfitting
- Binary cross-entropy loss
- Adam optimizer

## 📝 How to Use

1. Visit the deployed web app
2. Upload a PCB image (JPG/PNG format)
3. Wait for prediction (2-3 seconds)
4. View result: Defective ❌ or Undefective ✅

## 👨‍💻 Author

[Your Name]

## 📄 License

This project is for educational/industrial demonstration purposes.
