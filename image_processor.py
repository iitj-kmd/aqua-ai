import time
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# ------------------------------------------------
# LOAD DETR MODEL
# ------------------------------------------------

def load_detr_model():
    """
    Loads the facebook/detr-resnet-50 object detection model
    from Hugging Face transformers.
    """
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

# ------------------------------------------------
# DETECT OBJECTS IN IMAGE
# ------------------------------------------------

def detect_objects(pil_image, processor, model, threshold=0.3):
    """
    Takes a PIL image and returns a list of detected labels.

    Parameters:
    - pil_image: PIL.Image
    - processor: loaded DetrImageProcessor
    - model: loaded DetrForObjectDetection
    - threshold: float, minimum confidence score for detections

    Returns:
    - list of label names detected in the image
    """
    inputs = processor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )

    detected_labels = []
    if len(results[0]["boxes"]) > 0:
        for box, score, label in zip(
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"]
        ):
            label_text = model.config.id2label[label.item()]
            detected_labels.append(label_text)

    return detected_labels

# ------------------------------------------------
# DUMMY DB FUNCTION
# ------------------------------------------------

def dummy_send_to_db(info):
    """
    Simulates sending info to a database.
    """
    time.sleep(1)
    print("INFO SAVED TO DB:", info)
    return "Data sent to database (dummy)."
