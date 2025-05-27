import pathlib

import cv2
import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from sahi.predict import get_sliced_prediction
from torch import cuda

from napari_nuclephaser.utils import initialize_model

# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


@magic_factory(
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    ADVANCED_SETTINGS={},
    Generate_points={
        "tooltip": "If chosen, Points layer will be created with point at the center of bounding box for each detection"
    },
    Generate_bbox={
        "tooltip": "If chosen, Shapes layer will be created with rectangle representing bounding box of each detection"
    },
    Show_confidence={
        "tooltip": "If chosen, each rectangle in Shapes layer will have confidence score of each detection printed above it"
    },
    Confidence_threshold={
        "tooltip": "Parameter that determines how many detections will model return. Use calibration widgets to determine optimal threshold for your use case."
    },
    Sahi_size={
        "max": 100000,  # Default setting creates limit at 1000, this prevents it
        "tooltip": "Slicing window inference slice. The large image will be divided into small ones with this size in pixels. See obss/sahi library for more details",
    },
    Sahi_overlap={
        "tooltip": "Relative overlap between sliding windows. See obss/sahi library docs for more details."
    },
    Intersection_threshold={
        "tooltip": "A metric to determine when to detections are overlapping. If metric is higher than threshold, detections will be merged. See obss/sahi library docs for more details."
    },
    Points_size={
        "tooltip": "Points size in results Points layer. Can be changed later by pressing Ctrl+A and moving Size slider in the layer itself"
    },
    Bbox_thickness={
        "tooltip": "Thickness of the side of rectangles in Shapes layer if Generate bbox is chosen"
    },
    Score_text_size={
        "tooltip": "Font size of confidence score text if Show confidence parameter is chosen"
    },
    call_button="Predict",
    auto_call=False,
    result_widget=False,
)
def make_points(
    Select_image: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Confidence_threshold: float = 0.5,
    Generate_points=True,
    Generate_bbox=False,
    Show_confidence=False,
    ADVANCED_SETTINGS="",
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=10,
    Bbox_thickness=5,
    Score_text_size=3,
) -> napari.types.LayerDataTuple:
    """Takes a single-frame image of any size, YOLO object detection model (v5, v8 or v11) and SAHI parameters ->
    returns a detection in formats of Points layer and/or Shapes layer with boxes ahd corresponding confidence scores
    """
    pic = Select_image.data
    if (
        len(pic.shape) == 2
    ):  # Check if image is single channel. YOLO models work only with RGB images.
        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
    if len(pic.shape) > 3 or (
        len(pic.shape) == 3 and pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "Image is not a single frame! Choose different widget for processing stacks of images"
        )
        return None
    name = Select_image.name  # Fetch image name for further purposes
    if pic.dtype == np.uint16:
        pic = cv2.convertScaleAbs(pic, alpha=255 / 65535)
        pic = pic.astype(np.uint8)

    print("Initializing model...")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    print("Performing sliced prediction...")
    result = get_sliced_prediction(
        pic,
        detection_model,
        slice_height=Sahi_size,
        slice_width=Sahi_size,
        overlap_height_ratio=Sahi_overlap,
        overlap_width_ratio=Sahi_overlap,
        postprocess_type=Postprocess,
        postprocess_match_metric=Match_metric,
        postprocess_match_threshold=Intersection_threshold,
    )  # Standard SAHI sliced prediction code
    result = result.to_coco_predictions()
    print("Prediction is done!")

    def create_points(result):
        # Function for converting prediction results from COCO format into napari.layers.Points layer
        points = []
        for instance in result:
            bbox = instance["bbox"]
            Y, X = int(bbox[0] + (bbox[2] // 2)), int(bbox[1] + (bbox[3] // 2))
            points.append([X, Y])
        n_cells = len(points)
        points = np.array(points)

        viewer.add_points(
            points, size=Points_size, name=f"{n_cells} points {name}"
        )
        return points, n_cells

    def create_bbox(result):
        # Function for converting prediction results from COCO format into napari.layers.Shapes layer
        bboxes = []
        scores = []
        for instance in result:
            bbox = instance["bbox"]
            score = instance["score"]
            Y1, X1, Y2, X2 = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[0] + (bbox[2])),
                int(bbox[1] + (bbox[3])),
            )
            bboxes.append(np.array([[X1, Y1], [X1, Y2], [X2, Y2], [X2, Y1]]))
            scores.append(score)
        n_cells = len(scores)
        # bboxes, scores = np.array(bboxes), np.array(scores)

        # create the properties dictionary
        properties = {"score": scores}

        # specify the display parameters for the text

        if Show_confidence:
            text_parameters = {
                "string": "{score:.2f}",
                "size": Score_text_size,
                "color": "red",
                "anchor": "upper_left",
                "translation": [-3, 0],
            }

            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                text=text_parameters,
                name=f"{n_cells} bounding boxes {name}",
            )
        else:
            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                name=f"{n_cells} bounding boxes {name}",
            )
        return bboxes, scores, n_cells

    if Generate_points:
        print("Generating points...")
        create_points(result)
        print("Points are generated!")
    if Generate_bbox:
        print("Generating boxes...")
        create_bbox(result)
        print("Boxes are generated!")
    if not Generate_points and not Generate_bbox:
        print("None of the options are chosen, generating points as a default")
        create_points(result)
        print("Points are generated!")
    return None
