import numpy as np
import sahi.predict

from napari_nuclephaser import predict_on_single
from napari_nuclephaser.sahi_patch import custom_get_sliced_prediction


def test_patch_applied():
    """Verify that the SAHI get_sliced_prediction is replaced with our custom version."""
    assert sahi.predict.get_sliced_prediction is custom_get_sliced_prediction
    assert (
        sahi.predict.get_sliced_prediction.__name__
        == "custom_get_sliced_prediction"
    )


def test_widget_uses_custom_sliced_prediction():
    """
    Ensure that the widget's imported get_sliced_prediction is the same custom function.
    This confirms that when widgets call get_sliced_prediction, they use the patched version.
    """
    # The widget module imports get_sliced_prediction from sahi.predict
    # After the patch is applied, it should point to our custom function.
    assert (
        predict_on_single.get_sliced_prediction is custom_get_sliced_prediction
    )
    assert (
        predict_on_single.get_sliced_prediction.__name__
        == "custom_get_sliced_prediction"
    )


def test_custom_function_signature():
    """
    Quick test that the custom function accepts the new parameters and runs without error
    with a dummy model (mocked) to verify no syntax errors.
    """
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_model.confidence_threshold = 0.5
    mock_model.perform_batch_inference = MagicMock()
    mock_model.convert_original_predictions = MagicMock()
    mock_model.object_prediction_list_per_image = []

    with patch("sahi.predict.slice_image") as mock_slice:
        mock_slice.return_value = MagicMock(
            images=[],
            starting_pixels=[],
            original_image_height=100,
            original_image_width=100,
        )
        with patch(
            "sahi.predict.POSTPROCESS_NAME_TO_CLASS",
            new={"GREEDYNMM": MagicMock()},
        ):
            # Should not raise an exception
            result = custom_get_sliced_prediction(
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                detection_model=mock_model,
                filter_border_touching_detections=True,
                border_touching_width=5,
                progress_bar=False,
            )
            # Result is a PredictionResult; we just check it's created
            assert result is not None
