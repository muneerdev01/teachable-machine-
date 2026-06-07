import os
import io
from pathlib import Path

import streamlit as st
from PIL import Image

import backend.train_model as backend_train

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = backend_train.DATASET_DIR
MODEL_PATH = backend_train.MODEL_PATH

st.set_page_config(page_title="Teachable Machine UI", page_icon="🤖", layout="wide")

st.title("Teachable Machine Dashboard")
st.markdown(
    "Upload labeled images, train a model, and test new pictures from one dashboard."
)

# Dataset summary
st.sidebar.header("Dataset Summary")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)

class_dirs = [
    d for d in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]

if class_dirs:
    for class_name in class_dirs:
        count = len(
            [f for f in os.listdir(os.path.join(DATASET_DIR, class_name)) if not f.startswith(".")]
        )
        st.sidebar.write(f"- **{class_name}**: {count} images")
else:
    st.sidebar.write("No classes yet. Upload images to get started.")

# Upload new training samples
with st.expander("Upload new labeled samples"):
    with st.form(key="upload_form"):
        class_name = st.text_input("Class name", value="", help="Enter a label for this image class.")
        uploaded_files = st.file_uploader(
            "Select one or more images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        upload_button = st.form_submit_button("Upload Samples")

    if upload_button:
        if not class_name.strip():
            st.error("Please enter a class name.")
        elif not uploaded_files:
            st.error("Please choose one or more image files to upload.")
        else:
            class_folder = os.path.join(DATASET_DIR, class_name.strip())
            os.makedirs(class_folder, exist_ok=True)
            saved = 0

            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file)
                    ext = uploaded_file.name.split(".")[-1]
                    filename = f"{uploaded_file.name}".replace(" ", "_")
                    save_path = os.path.join(class_folder, filename)
                    image.save(save_path)
                    saved += 1
                except Exception as exc:
                    st.warning(f"Skipping file {uploaded_file.name}: {exc}")

            if saved:
                st.success(f"Uploaded {saved} images to class '{class_name.strip()}'")
            else:
                st.error("No files were saved.")

# Train model
with st.expander("Train model"):
    st.write("Train the image classification model using the current dataset.")
    if st.button("Train Model"):
        with st.spinner("Training model... this may take a few minutes on first run."):
            result = backend_train.train_model()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success("Training completed successfully.")
                st.write(f"Classes: {result.get('classes', [])}")
                st.write(f"Samples: {result.get('samples', 0)}")

# Prediction
with st.expander("Predict new images"):
    st.write("Upload an image and see the model's predicted class and confidence.")
    predict_files = st.file_uploader(
        "Select one or more images for prediction",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="predict_uploader",
    )

    if predict_files:
        model = backend_train.load_trained_model()
        if model is None:
            st.warning("No trained model found. Train the model first.")
        else:
            for upload in predict_files:
                try:
                    image = Image.open(upload).convert("RGB")
                    st.image(image, caption=upload.name, use_column_width=True)
                    predictions = backend_train.predict_image(model, image)
                    if not predictions:
                        st.error("Unable to predict this image.")
                        continue

                    top_class, top_prob = predictions[0]
                    st.markdown(f"**Prediction:** {top_class}")
                    st.markdown(f"**Confidence:** {top_prob * 100:.2f}%")

                    st.write("**All scores:**")
                    for class_name, prob in predictions:
                        st.write(f"- {class_name}: {prob * 100:.2f}%")
                except Exception as exc:
                    st.error(f"Failed to predict {upload.name}: {exc}")

# Helpful notes
st.markdown("---")
st.write(
    "Use the upload panel to add more pictures to each class, then train the model and test new images. "
    "The model is saved to the project `model/model.pkl` file."
)
