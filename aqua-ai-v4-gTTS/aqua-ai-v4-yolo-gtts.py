# main pythin file for the AquaSense app .
# This app tracks and monitors where the family members are consuming water as recommented or not .
# Developed by Kironmoy  Dhali (iitj-kmd)

import cv2
from ultralytics import YOLO
import time
from audio_manager import AudioThread
from utils import discard_first_50_frames

audio_thread = AudioThread()
audio_thread.start()

model = YOLO('../yolov8n.pt')

cap = cv2.VideoCapture(0)

desired_fps = 5
cap.set(cv2.CAP_PROP_FPS, desired_fps)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Desired FPS: {desired_fps}")
print(f"Actual FPS from camera: {actual_fps}")

skip_rate = int(actual_fps / desired_fps)

# class ids for the interested objects for this application
# in our application a person (0) and a bottle/cup(39/40)

PERSON_CLASS_ID = 0
BOTTLE_CLASS_ID = 39
CUP_CLASS_ID = 41


last_speech_time = 0
speech_interval = 15  # seconds

frame_idx = 0
FRAME_BATCH_SIZE = 10
frame_buffer = []

discard_first_50_frames(cap)

while cap.isOpened():

    success, frame = cap.read()
    frame_idx = frame_idx + 1
    if not success:
        break

    if len(frame_buffer) < FRAME_BATCH_SIZE:
        frame_buffer.append(frame)
        continue
    else:
        frame_buffer = []

    print(f'processing frame  ...{frame_idx}')
    person_detected = False
    water_object_detected = False

    results = model(frame, stream=True, verbose=False)

    # Process the  results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            # Check for a 'person'
            if label == 'person' and confidence > 0.5:
                person_detected = True

                # drawing bounding box around the person
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # check for a bottle
            elif (label == 'bottle' or label == 'cup') and confidence > 0.4:
                water_object_detected = True
                # draw bounding box around the water object
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            print(f'person_detected: {person_detected} water_object_detected: {water_object_detected}')

    if not water_object_detected:
        cv2.putText(frame, f'Where is water bottle?', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Implementing  the logic for the spoken message
    current_time = time.time()
    if person_detected and water_object_detected:
        if current_time - last_speech_time > speech_interval:
            message = "Okay, You are good... keep hydrated ..."
            audio_thread.say_message(message)
            last_speech_time = current_time
            print(message)
    elif person_detected and not water_object_detected:
        print(
            f'current_time - {current_time} - last_speech_time {last_speech_time} = {current_time - last_speech_time} ?  speech_interval {speech_interval}')
        if current_time - last_speech_time > speech_interval:
            message = "Where is your water bottle ?"
            print(message)
            audio_thread.say_message(message)
            last_speech_time = current_time

    # Display the frame
    cv2.imshow("Webcam Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
audio_thread.stop()
audio_thread.join()
cap.release()
cv2.destroyAllWindows()
