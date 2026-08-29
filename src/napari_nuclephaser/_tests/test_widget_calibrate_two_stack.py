from unittest.mock import MagicMock

import numpy as np
from napari.layers import Points

from napari_nuclephaser.predict_two_stack import predict_on_two_stack


def test_predict_on_two_stack_valid(make_napari_viewer, mocker, tmp_path):
    viewer = make_napari_viewer()
    # 2D stack: 2x3 frames of 50x50
    valid_stack = np.random.randint(0, 255, (2, 3, 50, 50), dtype=np.uint8)
    image_layer = viewer.add_image(valid_stack, name="test_2stack")

    mock_initialize = mocker.patch(
        "napari_nuclephaser.predict_two_stack.initialize_model"
    )
    mock_initialize.return_value = (MagicMock(), "mock_model")

    mock_sliced_pred = mocker.patch(
        "napari_nuclephaser.predict_two_stack.get_sliced_prediction"
    )
    mock_det1 = MagicMock()
    mock_det1.bbox.minx = 10
    mock_det1.bbox.maxx = 30  # 10+20
    mock_det1.bbox.miny = 10
    mock_det1.bbox.maxy = 30  # 10+20
    mock_det1.score.value = 0.9

    mock_det2 = MagicMock()
    mock_det2.bbox.minx = 30
    mock_det2.bbox.maxx = 50  # 30+20
    mock_det2.bbox.miny = 30
    mock_det2.bbox.maxy = 50  # 30+20
    mock_det2.score.value = 0.8

    mock_result = MagicMock()
    mock_result.object_prediction_list = [mock_det1, mock_det2]
    mock_sliced_pred.return_value = mock_result

    mock_show_info = mocker.patch(
        "napari_nuclephaser.predict_two_stack.show_info"
    )
    mock_show_error = mocker.patch(
        "napari_nuclephaser.predict_two_stack.show_error"
    )

    widget = predict_on_two_stack()
    widget(
        Select_stack=image_layer,
        viewer=viewer,
        Select_model="dummy_model_path",
        Detection_mode="Regular detection",
        Mode_file=None,
        Confidence_threshold=0.5,
        Save_result=False,
        Save_folder=tmp_path,
        Experiment_name="test_exp",
        Save_format="CSV",
    )

    # There are 2*3 = 6 frames, each with 2 detections → 12 points (4D)
    points_layer = viewer.layers[-1]
    assert isinstance(points_layer, Points)
    assert points_layer.data.shape == (12, 4), "Expected 12 points in 4D"

    mock_show_info.assert_called_once_with(
        "Made predictions for 2-stack successfully!"
    )
    mock_show_error.assert_not_called()
