def discard_first_50_frames(cap):
    first_frame_idx = 0
    while True:
        success, frame = cap.read()
        first_frame_idx = first_frame_idx + 1
        if first_frame_idx > 50:
            return
