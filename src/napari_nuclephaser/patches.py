import sahi.predict

from .sahi_patch import custom_get_sliced_prediction


def apply_sahi_patches():
    """Overrides SAHI's get_sliced_prediction with custom version."""
    sahi.predict.get_sliced_prediction = custom_get_sliced_prediction
