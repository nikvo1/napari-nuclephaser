import os

from sahi import AutoDetectionModel


def initialize_model(model_path, confidence_threshold, device):
    """Takes a YOLO model path, confidence threshold and device and returns an initialized model
    without explicitly passing model type"""

    model_type_list = ("yolov5", "ultralytics", "yolov8", "yolov11", "yolo11")

    for model_type in model_type_list:
        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            return (
                detection_model,
                model_type,
            )  # Return the successfully initialized model
        except TypeError:
            continue  # Continue to the next model type

    # Raise an error if all attempts fail
    raise RuntimeError(
        "Failed to initialize model from the provided file path."
    )


def create_unique_subfolder(parent_folder, subfolder_name):
    """Takes a root folder path and subfolder name and returns a straight up subfolder path if
    one doesn't exist, and returns same path with subfolder1/2/3... otherwise
    """
    base_path = os.path.join(parent_folder, subfolder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path

    counter = 1
    while True:
        new_name = f"{subfolder_name}{counter}"
        new_path = os.path.join(parent_folder, new_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        counter += 1
