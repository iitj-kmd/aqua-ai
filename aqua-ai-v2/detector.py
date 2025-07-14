from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch


def load_detr_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model


def detect_objects(image, processor, model, threshold=0.3):
    """
    image: numpy array (BGR)
    returns: boxes, labels, scores
    """
    image_rgb = image[..., ::-1]
    pil_img = Image.fromarray(image_rgb)

    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )

    boxes = results[0]["boxes"].tolist()
    labels = [model.config.id2label[label_id] for label_id in results[0]["labels"]]
    scores = results[0]["scores"].tolist()

    return boxes, labels, scores
