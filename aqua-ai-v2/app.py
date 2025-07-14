import streamlit as st
from streamlit_webrtc import webrtc_streamer
import video_processor

# ------------------------------------------------
# Streamlit Config
# ------------------------------------------------

st.set_page_config(
    page_title="Aqua-AI V2",
    layout="wide",
)

st.title("💧 Aqua-AI V2 — Smart Water Monitoring")

# ------------------------------------------------
# Initialize session state
# ------------------------------------------------

if "recognition_enabled" not in st.session_state:
    st.session_state.recognition_enabled = False

# ------------------------------------------------
# Sidebar controls
# ------------------------------------------------

st.sidebar.header("Controls")

sampling_rate = st.sidebar.slider(
    "Process every Nth frame",
    min_value=1,
    max_value=60,
    value=30,
    step=1,
)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶ Start Recognition"):
        st.session_state.recognition_enabled = True

with col2:
    if st.button("⏹ Stop Recognition"):
        st.session_state.recognition_enabled = False

# ------------------------------------------------
# Create a centered block for video
# ------------------------------------------------

st.markdown("""
    <style>
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 70%;
            height: auto;
            margin-top: 20px;
        }
        .video-container > div {
            width: 700px;
            max-width: 70%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="video-container">', unsafe_allow_html=True)

ctx = webrtc_streamer(
    key="aqua-ai-v2",
    video_processor_factory=lambda: video_processor.VideoProcessor(
        sampling_interval=sampling_rate
    ),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 800},
            "height": {"ideal": 600},
            "frameRate": {"ideal": 5, "max": 15},
        },
        "audio": False,
    },
    async_processing=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# Display recognition results
# ------------------------------------------------

if ctx.video_processor:
    labels = ctx.video_processor.last_detected_objects

    if labels:
        if "person" in labels and "bottle" in labels:
            st.success("✅ Both person and bottle detected! Stay hydrated!")
        else:
            missing = []
            if "person" not in labels:
                missing.append("Face (person)")
            if "bottle" not in labels:
                missing.append("Bottle")
            st.warning(f"⚠ Missing: {', '.join(missing)}. Please retake.")
    else:
        st.info("Recognition ready. Press ▶ Start Recognition to begin.")

else:
    st.info("Click 'Start' in the video box above to begin streaming.")
