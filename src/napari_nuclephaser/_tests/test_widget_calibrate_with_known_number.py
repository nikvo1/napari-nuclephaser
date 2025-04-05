import matplotlib

matplotlib.use("Agg")  # Set backend before other imports

import numpy as np

from napari_nuclephaser._widget import calibrate_with_known_number


def test_calibrate_with_known_number_happy_path(make_napari_viewer, mocker):
    # Setup viewer and add a test image layer
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(image_data)

    # Mock model initialization and SAHI prediction
    mock_initialize = mocker.patch(
        "napari_nuclephaser._widget.initialize_model"
    )
    mock_initialize.return_value = (mocker.MagicMock(), "MockModel")

    mock_get_sliced = mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction"
    )
    mock_result = mocker.MagicMock()
    # Mock three predictions with descending scores
    mock_result.to_coco_predictions.return_value = [
        {"score": 0.8},
        {"score": 0.7},
        {"score": 0.6},
    ]
    mock_get_sliced.return_value = mock_result

    # Instantiate the widget
    widget = calibrate_with_known_number()

    # Call the widget with test parameters
    result = widget(
        image_layer,
        viewer,
        Select_model="test_model.pt",
        Calibration_number=3,
        Postprocess="GREEDYNMM",
        Match_metric="IOS",
        Sahi_size=640,
        Sahi_overlap=0.2,
    )

    # Verify the returned threshold is correct
    expected = "Best threshold for model test_model.pt is 0.6"
    assert result == expected


def test_calibrate_with_image_stack_error(make_napari_viewer, mocker):
    # Setup viewer with a 3D image (stack)
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 256, (5, 100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(image_data)

    # Mock logger or error display if needed
    mock_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Instantiate the widget
    widget = calibrate_with_known_number()

    # Attempt calibration
    result = widget(image_layer, viewer)

    # Verify error handling
    mock_error.assert_called_with(
        "Image is not a single frame! Can't calibrate on a stack of images"
    )
    assert result is None
