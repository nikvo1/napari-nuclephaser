import cv2
import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Points


@magic_factory(
    auto_call=False,
    call_button="Convert",
    result_widget=False,
    Points_layer={"label": "Select points layer"},
    Reference_image={"label": "Select reference image"},
)
def convert_points_to_labels(
    Points_layer: Points,
    Reference_image: Image,
    viewer: napari.Viewer,
    Label_size=10,
):
    """Takes a Napari.Points layer (stacks are allowed) and reference image and returns points in Napari.Labels format.
    Main purpose is tracking with napari.btrack plugin (which accepts only Labels) or other plugins
    """
    points = Points_layer.data
    zeros = np.zeros(
        shape=(
            Reference_image.data.shape
            if len(Reference_image.data.shape) == 3
            else Reference_image.data.shape[:3]
        ),
        dtype=np.uint8,
    )
    for point in points:
        cv2.circle(
            zeros[int(point[0])],
            (int(point[2]), int(point[1])),
            Label_size,
            256,
            -1,
        )
    viewer.add_labels(
        zeros.astype(np.uint8),
        name=Points_layer.name + " labels",
        depiction="plane",
    )
