__version__ = "0.4.0"
# from .widget import widget_factory
# my_project/__init__.py
from .patches import apply_sahi_patches

# Apply the SAHI patch as soon as your package is imported.
apply_sahi_patches()
