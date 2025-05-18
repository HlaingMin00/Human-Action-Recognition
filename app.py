import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import tempfile

st.title("🧍‍♂️ Human Action Recognition with MoveNet + HAR Model")

# Load models only once
@st.cache_resource
def load_models():
    # Load MoveNet MultiPose from TensorFlow Hub
    movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet_model = movenet.signatures['serving_default']
    
    # Load custom HAR model
    har_model = tf.keras.models.load_model("harCNN+LSTM_3.h5")
    return movenet_model, har_model

movenet_model, har_model = load_models()

# Helper to run MoveNet
def detect_keypoints(image):
    input_img = tf.image.resize_with_pad(image, 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    input_img = tf.expand_dims(input_img, axis=0)

    outputs = movenet_model(input_img)
    keypoints = outputs["output_0"].numpy()
    return keypoints

# Helper to preprocess frame for HAR
def extract_pose_sequence(keypoints):
    # Simplify: Extract keypoint coordinates (exclude confidence for now)
    return keypoints[..., :2].reshape(1, -1, 2)

# User upload
uploaded_file = st.file_uploader("📷 Upload an image or video", type=["jpg", "png", "mp4", "mov"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        # Handle image input
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to tensor
        image_np = np.array(image)
        keypoints = detect_keypoints(tf.convert_to_tensor(image_np))

        pose_seq = extract_pose_sequence(keypoints)
        prediction = har_model.predict(pose_seq)
        predicted_class = np.argmax(prediction)

        st.success(f"🧠 Predicted Action Class: {predicted_class}")

    elif uploaded_file.type.startswith("video"):
        # Save uploaded video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = detect_keypoints(tf.convert_to_tensor(image_rgb))
            pose_seq = extract_pose_sequence(keypoints)
            prediction = har_model.predict(pose_seq)
            predicted_class = np.argmax(prediction)

            label = f"Action: {predicted_class}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
