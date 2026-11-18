import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

st.set_page_config(
    page_title="Cassava Leaf Disease Screening",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Cassava Leaf Disease Screening System")
st.write(
    "Upload a cassava leaf image to screen it for possible bacterial blight, "
    "brown streak disease, green mottle, mosaic disease, or a healthy leaf."
)
st.caption(
    "This tool gives a preliminary result only. It was trained on images "
    "from Uganda and has not yet been validated on Nigerian farms."
)

MODEL_PATH = "cassava_5class_finetuned.keras"

# This order must match the order the model learned during training.
CLASS_NAMES = [
    "Bacterial Blight",
    "Brown Streak Disease",
    "Green Mottle (associated with green mite damage)",
    "Healthy",
    "Mosaic Disease"
]

LOW_CONFIDENCE = 60.0  # percent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a cassava leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded cassava leaf image",
        use_container_width=True
    )

    if st.button("Analyse Leaf"):
        with st.spinner("Analysing leaf image..."):
            start_time = time.time()

            img = image.resize((224, 224))
            img_array = np.array(img).astype("float32")
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(prediction)) * 100
            response_time = time.time() - start_time

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.write(f"Response time: {response_time:.2f} seconds")

        st.write("All class scores:")
        for name, p in zip(CLASS_NAMES, prediction):
            st.write(f"- {name}: {p * 100:.1f}%")

        if confidence < LOW_CONFIDENCE:
            st.warning(
                "The model is not very sure about this image. Treat this "
                "result with extra caution and try a clearer photo."
            )

        if predicted_index == 3:
            st.warning(
                "A healthy result does not guarantee the leaf is free from "
                "disease or pest damage. If you still see unusual symptoms, "
                "please consult an agricultural extension officer."
            )
        else:
            st.warning(
                "Possible disease or pest damage detected. Please consult an "
                "agricultural extension officer or plant-health expert for "
                "confirmation."
            )
