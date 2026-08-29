from unittest.mock import MagicMock

import numpy as np
from napari.layers import Points

from napari_nuclephaser.predict_one_stack import predict_on_stack


def test_predict_on_stack_valid(make_napari_viewer, mocker, tmp_path):
    viewer = make_napari_viewer()
    valid_stack = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(valid_stack, name="test_stack")

    mock_initialize = mocker.patch(
        "napari_nuclephaser.predict_one_stack.initialize_model"
    )
    mock_detection_model = MagicMock()
    mock_initialize.return_value = (mock_detection_model, "mock_model")

    mock_sliced_pred = mocker.patch(
        "napari_nuclephaser.predict_one_stack.get_sliced_prediction"
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
        "napari_nuclephaser.predict_one_stack.show_info"
    )
    mock_show_error = mocker.patch(
        "napari_nuclephaser.predict_one_stack.show_error"
    )

    widget = predict_on_stack()
    widget(
        Select_stack=image_layer,
        viewer=viewer,
        Select_model="dummy_model_path",
        Detection_mode="Regular detection",
        Mode_file=None,
        Confidence_threshold=0.5,
        Postprocess="GREEDYNMM",
        Match_metric="IOS",
        Sahi_size=640,
        Sahi_overlap=0.2,
        Intersection_threshold=0.3,
        Points_size=30,
        Save_result=False,
        Save_folder=tmp_path,
        Experiment_name="test_exp",
        Save_format="CSV",
    )

    assert len(viewer.layers) == 2, "Points layer should be added"
    points_layer = viewer.layers[-1]
    assert isinstance(points_layer, Points), "Added layer should be Points"
    assert points_layer.data.shape == (20, 3), "Unexpected points shape"

    mock_show_info.assert_called_once_with(
        "Made predictions for stack successfully!"
    )
    mock_show_error.assert_not_called()


def test_predict_on_stack_invalid_input(make_napari_viewer, mocker):
    viewer = make_napari_viewer()
    invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(invalid_image, name="invalid_image")

    mock_show_error = mocker.patch(
        "napari_nuclephaser.predict_one_stack.show_error"
    )

    widget = predict_on_stack()
    widget(
        Select_stack=image_layer,
        viewer=viewer,
        Detection_mode="Regular detection",
    )

    mock_show_error.assert_called_once_with(
        "Chosen image is a single frame, not a stack!"
    )
    assert len(viewer.layers) == 1, "No additional layers should be added"


def test_predict_on_stack_saving_files(make_napari_viewer, mocker, tmp_path):
    viewer = make_napari_viewer()
    valid_stack = np.random.randint(0, 255, (2, 50, 50), dtype=np.uint8)
    image_layer = viewer.add_image(valid_stack, name="save_test")

    mock_initialize = mocker.patch(
        "napari_nuclephaser.predict_one_stack.initialize_model"
    )
    mock_initialize.return_value = (MagicMock(), "mock_model")

    mock_sliced_pred = mocker.patch(
        "napari_nuclephaser.predict_one_stack.get_sliced_prediction"
    )
    mock_result = MagicMock()
    mock_result.to_coco_predictions.return_value = []
    mock_sliced_pred.return_value = mock_result

    mocker.patch("napari_nuclephaser.predict_one_stack.show_info")
    mocker.patch("napari_nuclephaser.predict_one_stack.show_error")

    widget = predict_on_stack()
    widget(
        Select_stack=image_layer,
        viewer=viewer,
        Detection_mode="Regular detection",
        Mode_file=None,
        Save_result=True,
        Save_folder=tmp_path,
        Experiment_name="test_save",
        Save_format="CSV",
    )

    subfolder = tmp_path / "test_save"
    assert subfolder.exists(), "Subfolder should be created"

    csv_file = subfolder / "save_test count results.csv"
    metadata_file = subfolder / "save_test count metadata.txt"

    assert csv_file.exists(), "CSV file should be saved"
    assert metadata_file.exists(), "Metadata file should be saved"
