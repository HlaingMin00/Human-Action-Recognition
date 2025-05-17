import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import imageio

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    h, w, _ = frame.shape
    shaped = np.multiply(keypoints, [h, w, 1])

    # Dynamically scale radius based on image resolution
    radius = max(1, int(min(h, w) / 128))  # Tune the divisor for your preferred size

    for kp in shaped:
        y, x, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold=0.3):
    h, w, _ = frame.shape
    shaped = np.multiply(keypoints, [h, w, 1])

    # Dynamically scale line thickness based on image size
    relative_thickness = max(1, int(min(h, w) / 256))  # Adjust scaling base if needed

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (0, 0, 255),  # You can use color map here if desired
                     thickness=relative_thickness)

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
    cv2.putText(frame, f"People: {num_people}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

# Class names and previous prediction state
class_names = ["Standing", "Walking", "Running", "Sitting", "Falling"]
previous_states = []  # Stores [{'bbox': [...], 'pred_index': int}]
bbox_threshold = 0.1  # Distance threshold for bbox matching

def get_closest_person_index(current_bbox, tracked_bboxes):
    min_distance = float('inf')
    index = -1
    for i, tracked_bbox in enumerate(tracked_bboxes):
        dist = np.linalg.norm(np.array(current_bbox[:4]) - np.array(tracked_bbox[:4]))
        if dist < min_distance and dist < bbox_threshold:
            min_distance = dist
            index = i
    return index

def loop_through_people(org_image, frame, results, edges, confidence_threshold):
    global previous_states

    h, w, _ = org_image.shape
    num_people = 0
    keypoints_with_scores_conf = results["output_0"].numpy()[:, :, :56]
    new_previous_states = []

    for i, person_data in enumerate(keypoints_with_scores_conf[0]):
        person = person_data[:51].reshape(17, 3)
        bbox = person_data[51:]  # [ymin, xmin, ymax, xmax, score]

        if bbox[4] < confidence_threshold:
            continue

        num_people += 1
        draw_connections(org_image, person, edges, confidence_threshold)
        draw_keypoints(org_image, person, confidence_threshold)

        model_input = person[:, :2].flatten().reshape(1, 34)
        prediction = model.predict(model_input)
        pred_index = int(np.argmax(prediction))

        match_index = get_closest_person_index(bbox, [p['bbox'] for p in previous_states])
        if match_index != -1:
            previous_index = previous_states[match_index]['pred_index']
            if previous_index == pred_index:
                display_index = pred_index
            else:
                display_index = previous_index
        else:
            display_index = pred_index

        new_previous_states.append({'bbox': bbox[:4], 'pred_index': display_index})
        action_label = class_names[display_index]

        if bbox[4] > confidence_threshold:
            ymin = int(bbox[0] * h)
            xmin = int(bbox[1] * w)
            ymax = int(bbox[2] * h)
            xmax = int(bbox[3] * w)

            cv2.rectangle(org_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            box_height = ymax - ymin
            font_scale = max(0.7, min(3, box_height / 150))
            thickness = max(2, int(font_scale * 1.5))

            label_x = xmin
            label_y = max(ymin - 10, 15)
            label_pos = (label_x, label_y)

            cv2.putText(org_image, action_label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    previous_states = new_previous_states
    draw_action_summary(org_image , num_people)

def resize_to_square_with_padding(img, divisible_by=32):
    h, w, _ = img.shape
    max_dim = max(h, w)

    # Automatically choose square size rounded up to nearest multiple of 32
    square_size = ((max_dim + divisible_by - 1) // divisible_by) * divisible_by

    # Calculate padding to center the image in the square
    pad_vert = square_size - h
    pad_horiz = square_size - w
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    # Pad image to make it square
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # No need to resize since square_size = padded size
    return padded_img, (pad_left, pad_top), (w, h), square_size

# import keras
# model = keras.models.load_model('/content/harCNN+LSTM_3.h5')
from tensorflow.keras.models import load_model
model = load_model('harCNN+LSTM_3.h5')

def uploadPhoto():
    image = st.file_uploader("Upload the file")
    if image is not None:
        st.success("Photo was successfully uploaded!")
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        org_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(org_image, channels="BGR")
        return org_image

def takePhoto():
    image = st.camera_input("Please take a photo")
    if image is not None:
        st.success("Photo was successfully taken!")
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        org_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # st.image(org_image, channels="BGR")
        return org_image
    
def detect(org_image):
    # Detection section
    frame = tf.image.resize_with_pad(tf.expand_dims(org_image, axis=0),256,256)
    frame = np.array(frame, dtype=np.uint8)
    input_img = tf.cast(frame, dtype=tf.int32)
    results = movenet(input_img)

    resized_img,_,_,_ = resize_to_square_with_padding(org_image)
    loop_through_people(resized_img,frame, results, EDGES, 0.1)

    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption="Multiperson Action Recognition")

def uploadvideo(video_file):
    if video_file is None:
        return None

    try:
        # Save the uploaded file to disk
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_temp.write(video_file.read())
        input_temp.flush()
        input_path = input_temp.name

        cap = cv2.VideoCapture(input_path)
        ret, first_frame = cap.read()
        if not ret:
            st.error("Could not read the first frame.")
            cap.release()
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 24  # fallback default

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a temp output file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, org_frame = cap.read()
            if not ret:
                break

            # TensorFlow preprocessing
            frame = tf.image.resize_with_pad(tf.expand_dims(org_frame, axis=0), 256, 256)
            frame = np.array(frame, dtype=np.uint8)
            input_img = tf.cast(frame, dtype=tf.int32)
            results = movenet(input_img)
            frame = np.squeeze(frame)

            # Draw keypoints
            resized_img, _, _, _ = resize_to_square_with_padding(org_frame)
            loop_through_people(resized_img, frame, results, EDGES, 0.1)

            # Ensure format for imageio
            resized_img = resized_img.astype(np.uint8)
            if resized_img.shape[2] == 3:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

            writer.append_data(resized_img)

        cap.release()
        writer.close()

        return output_path

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None


def contact():
    st.write('Write to the developer')
    form_submit = """<form action="https://formsubmit.co/hlaingminoo4224@gmail.com" method="POST">
     <input type="text" name="name" placeholder=" 🙍🏽‍♂️ Name "required>
     <input type="email" name="email" placeholder=" ✉️ Email Address">
     <textarea id="subject" name="subject" placeholder=" 📝 Write something here..." style="height:200px"></textarea>
     <input type="hidden" name="_captcha" value="false">
     <button type="submit">Send</button>
     </form>
     <style>
input[type=text],input[type=email], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
  margin-top: 6px;
  margin-bottom: 16px;
  resize: vertical;
}
button[type=submit] 
{
  background-color: #D1E5F3;
  color: black;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
button[type=submit]:hover
{
  background-color: #2E34DA;
  color = white;
}
</style>
     """
    st.markdown(form_submit,unsafe_allow_html=True)
#     components.html(form_submit, height=500)

def main():
    menu = ['Home Page', 'Contact developer']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home Page':
        st.header('Human Action Recognition')
        upload_option = st.sidebar.selectbox("Photo Options", ('Upload Image📁', 'Shoot Photo📷', 'Upload Video'))

        # ====== Image Upload ======
        if upload_option == 'Upload Image📁':
            image = uploadPhoto()
            if st.button("Detect"):
                if image is None:
                    st.warning("Please upload or shoot a photo before classifying.")
                else:
                    with st.spinner("Detecting..."):
                        detect(image)

        # ====== Webcam Photo ======
        elif upload_option == 'Shoot Photo📷':
            image = takePhoto()
            if st.button("Detect"):
                if image is None:
                    st.warning("Please upload or shoot a photo before classifying.")
                else:
                    with st.spinner("Detecting..."):
                        detect(image)

        # ====== Video Upload ======
        elif upload_option == 'Upload Video':
            video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
            if video_file and st.button("Process Video"):
                with st.spinner("Processing..."):
                    video_path = uploadvideo(video_file)
                    if video_path:
                        st.success("Video created!")
                        with open(video_path, "rb") as f:
                            st.video(f.read(), format="video/mp4")
                        os.remove(video_path)
                    else:
                        st.error("Failed to process video.")


    elif choice == 'Contact developer':
        contact()

if __name__ =='__main__':
    main()
