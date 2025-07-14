from streamlit_webrtc import VideoProcessorBase
import av
import streamlit as st

import detector
import utils


class VideoProcessor(VideoProcessorBase):
    def __init__(self, sampling_interval=30):
        self.frame_count = 0
        self.sampling_interval = sampling_interval

        # Load the model once
        self.processor, self.model = detector.load_detr_model()

        # Latest detected labels
        self.last_detected_objects = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Check if recognition is enabled
        recognition = st.session_state.get("recognition_enabled", False)

        if recognition:
            self.frame_count += 1
            if self.frame_count % self.sampling_interval == 0:
                boxes, labels, scores = detector.detect_objects(
                    img, self.processor, self.model
                )
                self.last_detected_objects = labels
                img = utils.draw_boxes(img, boxes, labels, scores)
                print("detecting objects ..")
        else:
            # Not running detection
            self.last_detected_objects = []

        return av.VideoFrame.from_ndarray(img, format="bgr24")
