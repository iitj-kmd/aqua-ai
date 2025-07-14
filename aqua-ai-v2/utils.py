import cv2


def draw_boxes(image, boxes, labels, scores, color=(0, 255, 0)):
    """
    Draw bounding boxes on the image.

    Parameters:
    - image: numpy array (BGR)
    - boxes: list of [xmin, ymin, xmax, ymax]
    - labels: list of label strings
    - scores: list of confidence scores
    """
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = [int(v) for v in box]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - 10),
            (x_min + text_width, y_min),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image
