import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time

import image_processor

# ------------------------------------------------
# LOAD DETR MODEL (cached)
# ------------------------------------------------

@st.cache_resource
def load_model_once():
    return image_processor.load_detr_model()

processor_obj, model_obj = load_model_once()

# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Aqua-AI",
    layout="wide",
)

# Sidebar
st.sidebar.title("Aqua-AI Sidebar")
st.sidebar.write("Settings and options will go here.")

st.title("Aqua-AI - Smart Water Monitoring")

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------

if "capture_enabled" not in st.session_state:
    st.session_state.capture_enabled = True

if "retake_enabled" not in st.session_state:
    st.session_state.retake_enabled = False

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# ------------------------------------------------
# UI PLACEHOLDERS
# ------------------------------------------------

frame_placeholder = st.empty()
info_bar = st.empty()

# Initialize webcam
#camera = cv2.VideoCapture(0)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



# ------------------------------------------------
# CAPTURE FRAME LOOP
# ------------------------------------------------

ret, frame = camera.read()
if not ret:
    info_bar.error("Failed to read frame from webcam.")
else:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Keep last frame in memory for processing
    st.session_state.last_frame = frame_rgb

    # Display the video feed
    frame_placeholder.image(frame_rgb, channels="RGB")

# ------------------------------------------------
# BUTTONS (STATIC DECLARATION)
# ------------------------------------------------

# We declare buttons once.
col1, col2 = st.columns([1, 1])

with col1:
    capture_clicked = st.button(
        "Capture",
        disabled=not st.session_state.capture_enabled,
        key="capture_button"
    )

with col2:
    retake_clicked = st.button(
        "Retake",
        disabled=not st.session_state.retake_enabled,
        key="retake_button"
    )

# ------------------------------------------------
# PROCESS CAPTURE BUTTON
# ------------------------------------------------

if capture_clicked:
    if st.session_state.last_frame is not None:
        pil_img = Image.fromarray(st.session_state.last_frame)

        detected_objects = image_processor.detect_objects(
            pil_img,
            processor_obj,
            model_obj,
            threshold=0.3
        )

        person_found = "person" in detected_objects
        bottle_found = "bottle" in detected_objects

        missing = []
        if not person_found:
            missing.append("Face (person)")
        if not bottle_found:
            missing.append("Bottle")

        if missing:
            # Show error and enable retake
            info_bar.error(
                f"⚠ Missing items: {', '.join(missing)}. Please retake the photo."
            )
            st.session_state.capture_enabled = False
            st.session_state.retake_enabled = True
        else:
            # Both objects found → success
            info_bar.success(
                "✅ Thanks for your AquaSense! Have a nice hydrated day!"
            )
            image_processor.dummy_send_to_db({
                "detected_objects": detected_objects,
                "timestamp": time.time(),
            })
            st.session_state.capture_enabled = True
            st.session_state.retake_enabled = False
    else:
        info_bar.error("No frame available to capture.")

# ------------------------------------------------
# PROCESS RETAKE BUTTON
# ------------------------------------------------

if retake_clicked:
    # Reset state
    st.session_state.capture_enabled = True
    st.session_state.retake_enabled = False
    info_bar.info("Retake initiated. Ready for new capture.")
