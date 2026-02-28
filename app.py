import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("best_model.h5", compile=False)

class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus (Normal Skin)",
    "Vascular Lesion"
]

cancer_classes = [0, 1, 4]

st.title("Skin Cancer Detection System")

uploaded = st.file_uploader("Upload Skin Image", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224,224))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    skin_type = class_names[idx]

    if idx in cancer_classes:
        status = "⚠ Cancer Detected"
        cancer_type = skin_type
    else:
        status = "✅ Non-Cancerous"
        cancer_type = None

    st.subheader("Prediction Result")
    st.write("Skin Type:", skin_type)
    st.write("Status:", status)

    if cancer_type:
        st.write("Cancer Type:", cancer_type)

    st.write(f"Confidence: {confidence:.2f}%")