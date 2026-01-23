import streamlit as st
import requests
from PIL import Image
import time

# --- Configuration ---
API_URL = "https://backend-168439508542.europe-west1.run.app/predict"

st.set_page_config(
    page_title="Real vs Fake Image Classifier",
    layout="centered",
)

st.title("Real vs Fake Image Classification")
st.write("Upload an image to check it is **real** or **AI-generated**.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            start_time = time.perf_counter()

            try:
                files = {
                    "data": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(API_URL, files=files, timeout=30)
                elapsed = time.perf_counter() - start_time

                if response.status_code == 200:
                    result = response.json()
                    probability = result["probability"]
                    prediction = result["prediction"]

                    st.success(f"### Prediction: **{prediction}**")
                    st.metric(
                        label="Fake Image Probability",
                        value=f"{probability:.2%}",
                    )
                    st.caption(f"Inference time: {elapsed:.3f} seconds")

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
