import pathlib

import matplotlib
from magicgui import magic_factory
from napari.layers import Points
from torch import cuda

matplotlib.use("Agg")
# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("yolov5", "ultralytics", "yolov8", "yolov11", "yolo11")


@magic_factory(
    auto_call=False,
    call_button="Count Up",
    result_widget=True,
    Points_layer={"label": "Select points layer"},
)
def give_num_points(Points_layer: Points):
    # Count up the points from the label layer
    points = Points_layer
    return len(points.data)
