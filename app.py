import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
import av
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Autonomous Parking System", layout="wide")
st.title("Autonomous Parking Detection")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
mode = st.sidebar.selectbox("Choose Input Mode", ["Image", "Video", "Webcam"])
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

class YOLOVideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(source=img, conf=conf, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width="stretch")

        with col2:
            results = model.predict(source=image_np, conf=conf)
            plotted = results[0].plot()
            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, caption="Detected Output", width="stretch")

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()


elif mode == "Webcam":
    st.write("Start webcam for real-time parking detection.")
    webrtc_streamer(
        key="parking-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
