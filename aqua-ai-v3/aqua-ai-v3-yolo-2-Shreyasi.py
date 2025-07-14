import cv2
from ultralytics import YOLO
import pyttsx3
import time
# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load a pre-trained YOLO model
# You can choose different models like 'yolov8n.pt' (nano), 'yolov8s.pt' (small), etc.
# The 'n' model is lighter and faster, good for real-time applications.
model = YOLO('../yolov8n.pt')

# Open the laptop webcam
cap = cv2.VideoCapture(0)
# Set the desired FPS
desired_fps = 3
cap.set(cv2.CAP_PROP_FPS, desired_fps)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Get the actual FPS from the camera (it might be different)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Desired FPS: {desired_fps}")
print(f"Actual FPS from camera: {actual_fps}")

# Define the classes we are interested in
# These correspond to the COCO dataset classes
PERSON_CLASS_ID = 0
BOTTLE_CLASS_ID = 39 # 'bottle' class
CUP_CLASS_ID = 41    # 'cup' class

# Keep track of the last time a message was spoken to avoid repeating too often
last_speech_time = 0
speech_interval = 5 # seconds

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        break
    last_processed_time_1=0
    current_time_1 = time.time()
    if current_time_1 - last_processed_time_1 <= 1.0:
        continue

    last_processed_time_1 = current_time_1

    person_detected = False
    water_object_detected = False

    # Run YOLO object detection on the frame
    results = model(frame, stream=True)


    # Process the detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            # Check for a 'person'
            if label == 'person' and confidence > 0.3:
                person_detected = True
                # Draw bounding box around the person
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check for a 'bottle' or 'cup'
            elif (label == 'bottle' or label == 'cup') and confidence > 0.3:
                water_object_detected = True
                # Draw bounding box around the water object
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Implement the logic for the spoken message
    current_time = time.time()
    if person_detected and water_object_detected:
        if current_time - last_speech_time > speech_interval:
            message = "Okay, You are good to go honey ... keep hydrated ..."
            engine.say(message)
            engine.runAndWait()
            last_speech_time = current_time
            print(message)
    elif person_detected and not water_object_detected:
        if current_time - last_speech_time > speech_interval:
            message = "Honey get a water bottle please"
            engine.say(message)
            engine.runAndWait()
            last_speech_time = current_time
            print(message)
        # Optional: You can add a different message here if you want
        # For simplicity, we'll keep the logic as you described

    # Display the resulting frame
    cv2.imshow("Webcam Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()