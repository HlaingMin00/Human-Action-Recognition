import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import cv2
import tensorflow as tf

# Page configuration: title and layout
st.set_page_config(page_title="Multi-Person HAR", layout="wide")

# 1. Load and cache models at startup (so they load only once):contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
@st.cache_resource
def load_models():
    """
    Load MoveNet MultiPose and HAR models (from files or TF Hub).
    Cached as a global resource to avoid reloading on each rerun.
    """
    # Load MoveNet MultiPose TFLite interpreter
    movenet_path = "movenet_multi_pose.tflite"  # Path to your MoveNet model file
    interpreter = tf.lite.Interpreter(model_path=movenet_path)
    interpreter.allocate_tensors()
    # Load the custom HAR Keras model
    har_model_path = "custom_har_model.h5"  # Path to your HAR model file
    har_model = tf.keras.models.load_model(har_model_path)
    return interpreter, har_model

# Functions for detection
def detect_keypoints(interpreter, image_np):
    """
    Run MoveNet MultiPose on a numpy RGB image.
    Returns keypoints with scores for each detected person.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Determine model input size and preprocess image
    input_size = input_details[0]['shape'][1]  # e.g., 256 for MoveNet Lightning
    img_resized = cv2.resize(image_np, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores  # shape: [1, num_persons, 56]

def predict_actions(keypoints_with_scores, har_model):
    """
    Predict actions for each person given their keypoints.
    Returns a list of action indices.
    """
    # Example: flatten keypoints for each person and run through HAR model
    num_persons = keypoints_with_scores.shape[1]
    actions = []
    for i in range(num_persons):
        # Extract one person's keypoints (17 points * (x,y,score) = 51 values)
        person_kp = keypoints_with_scores[0, i, :].reshape(-1, 17*3)
        # Predict action (assumes model input matches this format)
        pred = har_model.predict(person_kp)
        action_idx = int(np.argmax(pred, axis=1)[0])
        actions.append(action_idx)
    return actions

# Initialize models
interpreter, har_model = load_models()

# App title
st.title("Multi-Person Human Action Recognition")

# Use tabs to separate Image, Webcam, and Video modes
tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam Capture", "Video Processing"])

with tab1:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read image and display
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        st.image(image, caption="Input Image", use_column_width=True)
        # Run pose and action detection
        with st.spinner("Detecting poses and actions..."):
            keypoints = detect_keypoints(interpreter, image_np)
            actions = predict_actions(keypoints, har_model)
        # Annotate image (draw keypoints and action labels) - pseudo-code
        annotated = image_np.copy()
        # (You would draw circles/lines for keypoints and put action text here)
        # For example: cv2.putText(annotated, f"Action {a}", (x, y), ... )
        st.image(annotated, caption="Detected Poses and Actions", use_column_width=True)
        st.write(f"Detected actions for {len(actions)} person(s): {actions}")

with tab2:
    st.header("Webcam Capture")
    cam_img = st.camera_input("Take a picture")
    if cam_img:
        image = Image.open(cam_img).convert('RGB')
        image_np = np.array(image)
        st.image(image, caption="Webcam Image", use_column_width=True)
        with st.spinner("Detecting poses and actions..."):
            keypoints = detect_keypoints(interpreter, image_np)
            actions = predict_actions(keypoints, har_model)
        annotated = image_np.copy()
        # (Draw keypoints/action labels here)
        st.image(annotated, caption="Detected Poses and Actions", use_column_width=True)
        st.write(f"Detected actions for {len(actions)} person(s): {actions}")

with tab3:
    st.header("Video Processing (Advanced)")
    st.write("Upload a video (MP4/AVI) for action recognition. This may take time.")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if video_file:
        # Save uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        cap = cv2.VideoCapture(temp_file.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Video loaded: {frame_count} frames.")
        with st.spinner("Processing video frames..."):
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # Run pose and action detection per frame
                keypoints = detect_keypoints(interpreter, frame)
                actions = predict_actions(keypoints, har_model)
                # (Optional: annotate frame with results and optionally display or save)
                # We skip displaying each frame for performance.
            cap.release()
        st.success(f"Video processed ({frame_idx} frames).")
        # (Optional: display last frame or summary if needed)
