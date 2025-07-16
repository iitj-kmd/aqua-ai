# 💧 Aqua-AI 

Aqua-AI V2 is an AI-powered web application built with Streamlit that helps intelligently monitor water consumption by detecting whether a person is holding a water bottle in front of a webcam.

The app uses a machine learning model from Hugging Face called **DETR (DEtection TRansformer)** to detect objects in a video stream in real-time. It highlights detected objects like a person and a bottle with bounding boxes.


**Aqua-AI V4** is a new version of the Aqua-AI project that uses a **native YOLO object detection model** built entirely with OpenCV and PyTorch — instead of relying on Hugging Face models.

This version detects whether a person is holding a water bottle in front of a webcam, helping promote healthy water consumption habits.

Note :  All versions are in their own folder
---

## 🎯 Key Differences from V2

✅ Uses **native YOLO** (e.g. YOLOv5 or YOLOv8)  
✅ Runs locally without internet after installing weights  
✅ Faster inference speed compared to large transformers  
✅ Smooth video which is not attainable in streamlit UI
✅ Fully integrated with:
- **OpenCV** for image handling and bounding boxes
- **PyTorch** for running YOLO model
- 
---

## Project Purpose

✅ Monitor hydration habits  for family members 
✅ Encourage people to drink water regularly

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
