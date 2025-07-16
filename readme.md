# 💧 Aqua-AI V2

Aqua-AI V2 is an AI-powered web application built with Streamlit that helps intelligently monitor water consumption by detecting whether a person is holding a water bottle in front of a webcam.

The app uses a machine learning model from Hugging Face called **DETR (DEtection TRansformer)** to detect objects in a video stream in real-time. It highlights detected objects like a person and a bottle with bounding boxes.

---

## Project Purpose

✅ Monitor hydration habits  
✅ Encourage people to drink water regularly  
✅ Practice computer vision and machine learning techniques  
✅ Learn how to integrate real-time video streaming in a web app

---

## Features

- Live webcam video stream
- Detects:
  - **person** (as a proxy for the face)
  - **bottle** (water bottle)
- Draws bounding boxes around detected objects
- User can:
  - Start or stop recognition at any time
  - Adjust how often frames are processed for better speed
- Centered and enlarged video window
- Simple user interface
- Built with:
  - **Streamlit**
  - **streamlit-webrtc**
  - **OpenCV**
  - **Hugging Face Transformers (DETR model)**

---

## Installation

First, make sure you have **Python 3.11 or above** installed.

Then install the required Python packages:

```bash
pip install streamlit streamlit-webrtc transformers torch pillow opencv-python av
