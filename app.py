import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import tempfile
import imageio
import os
import ffmpeg

st.title("🧍‍♂️ Human Action Recognition with MoveNet + HAR Model")

# Load models only once
@st.cache_resource
def load_models():
    # Load MoveNet MultiPose from TensorFlow Hub
    movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet_model = movenet.signatures['serving_default']
    
    # Load custom HAR model
    har_model = tf.keras.models.load_model("CNN+LSTM_002.h5")
    return movenet_model, har_model

movenet_model, har_model = load_models()

def resize_with_pad(image, target_size=512, pad_color=(255, 255, 255)):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding calculations
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Pad and return
    padded_image = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded_image

# Helper to run MoveNet
def detect_keypoints(image):
    input_img = tf.image.resize_with_pad(image, 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    input_img = tf.expand_dims(input_img, axis=0)

    outputs = movenet_model(input_img)
    keypoints = outputs["output_0"].numpy()
    return keypoints

def draw_action_summary(frame, num_people):
    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    # Define box size proportional to frame size
    box_width = int(frame_width * 0.2)  # 20% of frame width
    box_height = int(frame_height * 0.05)  # 5% of frame height

    # Set fixed padding & position (top-left corner)
    padding = int(box_height * 0.2)
    box_x, box_y = padding, padding

    # Dynamically adjust text size
    font_scale = min(frame_width, frame_height) * 0.001  # Scale based on frame size
    text_thickness = max(1, int(font_scale * 3))  # Ensure minimum thickness

    # Draw background rectangle
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)

    # Position text inside the box
    text_x = box_x + padding
    text_y = box_y + int(box_height * 0.7)  # Centered vertically in box

    # Display number of people detected
    cv2.putText(frame, f"Person: {num_people}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

# Store previous class index per person
# Track per-person state
last_class_index = {}
pending_class_index = {}
repeat_count = {}

# Tracking state
person_tracker = {}
next_person_id = 0

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_stable_action(i, current_index, required_repeats=5):
    """Handles action stability for a given person ID (i) based on current prediction."""
    if i not in last_class_index:
        last_class_index[i] = current_index
        pending_class_index[i] = current_index
        repeat_count[i] = 0
        return current_index
    if current_index == last_class_index[i]:
        # Same as last confirmed → reset
        repeat_count[i] = 0
        pending_class_index[i] = current_index
        return current_index
    if current_index == pending_class_index[i]:
        # Second time seeing this new class
        repeat_count[i] += 1
        if repeat_count[i] >= required_repeats:
            last_class_index[i] = current_index
            repeat_count[i] = 0
        return last_class_index[i]
    # New class seen first time
    pending_class_index[i] = current_index
    repeat_count[i] = 1
    return last_class_index[i]

current_frame_index =0
last_seen = {}
class_names = ["Standing", "Walking", "Running", "Sitting", "Falling"]
def har_on_person(image, keypoints, confidence_threshold=0.1):
    global last_class_index, person_tracker, next_person_id,current_frame_index,last_seen
    h, w, _ = image.shape
    num_people = 0
    new_tracker = {}

    for i, person_data in enumerate(keypoints[0]):
        bbox = person_data[51:]
        conf = person_data[:51].reshape(17, 3)[:, 2]
        if bbox[4] < confidence_threshold or np.mean(conf) < 0.1:
            continue
        num_people += 1
        ymin = int(bbox[0] * h)
        xmin = int(bbox[1] * w)
        ymax = int(bbox[2] * h)
        xmax = int(bbox[3] * w)
        current_box = [xmin, ymin, xmax, ymax]

        # Match with existing tracked boxes
        matched_id = None
        for pid, prev_box in person_tracker.items():
            if get_iou(current_box, prev_box) > 0.3:
                matched_id = pid
                break
        if matched_id is None:
            matched_id = next_person_id
            next_person_id += 1

        new_tracker[matched_id] = current_box
        last_seen[matched_id] = current_frame_index  # 🔁 track last seen frame

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        model_input = person_data[:51].reshape(17, 3)[:, :2].flatten().reshape(1, 34)
        prediction = har_model.predict(model_input)
        current_index = int(np.argmax(prediction))
        box_height = ymax - ymin
        font_scale = max(0.7, min(1, box_height / 150))
        thickness = max(2, int(font_scale * 1.5))
        label_x = xmin
        label_y = max(ymin - 10, 15)
        label_pos = (label_x, label_y)
        action_index = get_stable_action(matched_id, current_index)
        label = f"{class_names[action_index]}" #Person:{matched_id} ,for person tracking prove
        cv2.putText(image, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    person_tracker = new_tracker
    draw_action_summary(image, num_people)


# User upload
uploaded_file = st.file_uploader("📷 Upload an image or video", type=["jpg", "png", "mp4", "mov"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        # Handle image input
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        # Convert image to tensor
        image_np = np.array(image)
        keypoints = detect_keypoints(tf.convert_to_tensor(image_np))
        image_np = resize_with_pad(image_np)
        har_on_person(image_np,keypoints)
        st.image(image_np, caption="Multiperson Action Recognition")

    elif uploaded_file.type.startswith("video"):
        # Save uploaded video to a temp file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        writer = imageio.get_writer(output_path, fps=27, codec='libx264', quality=8)

        # Save uploaded video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        frame_count=0
        if st.button("Process Video"):
            with st.spinner("Processing..."):
                cap = cv2.VideoCapture(tfile.name)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image_rgb= np.array(frame)
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                    keypoints = detect_keypoints(tf.convert_to_tensor(image_rgb))
                    image = resize_with_pad(image_rgb)
                    har_on_person(image,keypoints)
                    writer.append_data(image)
            cap.release()
            writer.close()
            if output_path:
                st.session_state.video_path = output_path
                st.session_state.video_ready = True
            else:
                st.error("Failed to process video.")
    
        # Show the video if it's ready
        if st.session_state.get("video_ready") and "video_path" in st.session_state:
            st.success("Video created!")
            with open(st.session_state.video_path, "rb") as f:
                st.video(f.read(), format="video/mp4")
    
            # Button to clear everything and reset
            if st.button("🧹 Clear Everything"):
                try:
                    os.remove(st.session_state.output_path)
                except Exception as e:
                    st.warning(f"Could not remove file: {e}")
                for key in ["video_path", "video_ready"]:
                    st.session_state.pop(key, None)
                st.rerun()

st.markdown(
    """
    <a href="https://huggingface.co/spaces/hlaingmin00/har-gradio-demo" target="_blank">
        <button style='padding:10px 20px;font-size:16px;background-color:#4CAF50;color:white;border:none;border-radius:5px;'>
            Real time Multi-person Human Action Recognition
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

