Predict on single image
=======================

Description
+++++++++++

This widget is used for running prediction with :doc:`YOLO </General information/Object detection overview>` and :doc:`SAHI </General information/Sliced inference overview>` on a single image.
An image can be of any size (no lower or upper limit) and any format supported by Napari (RGB or single channel, 8‑bit or 16‑bit).

The widget can run in three **Detection modes**:
  - **Regular detection** – uses a fixed confidence threshold.
  - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` – uses test‑time augmentations (requires a metadata file from :doc:`Calibrate with points </Widgets/Calibrate with points>`).
  - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` – uses a dynamic threshold model (requires a `.pkl` file from :doc:`Calibrate with dynamic threshold </Widgets/Calibrate dynamic threshold>`).

.. figure:: ../Images/Predict_single_image.jpg
        :scale: 30 %
        :align: center
        :alt: The image didn't load(

        Predict on single image widget results. It returns bounding boxes and confidence scores, as well as number of objects in the name of the result layer.

Parameters
++++++++++

**Select image** – the image to run inference on. Only single images are accepted; stacks will raise an error.

**Select model** – the YOLO model to use for inference. Only small models (n and s) are automatically downloaded; larger models can be downloaded from the `NuclePhaser GitHub page <https://github.com/nikvo1/napari-nuclephaser>`_.

**Detection mode** – choose between:
  - Regular detection
  - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` (requires a `.txt` metadata file)
  - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` (requires a `.pkl` model file)

**Mode file** – the file needed for TTA or dynamic threshold modes (`.txt` for TTA, `.pkl` for dynamic threshold). This field is only active when the corresponding mode is selected.

**Output format** – choose the type of layer to add:
  - Points (centered and bounding boxes centers)
  - Bounding boxes
  - Bounding boxes with confidence scores

**Confidence threshold** – the threshold used in **Regular detection** mode. This is the most important parameter for counting accuracy; see :doc:`Confidence threshold calibration </General information/Confidence threshold calibration>`.

**SAHI parameters** – these are advanced settings; see :doc:`Sliced inference overview </General information/Sliced inference overview>` for details:
  - **Sahi size** – sliding window size in pixels.
  - **Sahi overlap** – relative overlap between windows.
  - **Postprocess** – algorithm to merge overlapping detections (GREEDYNMM, NMS, NMM).
  - **Match metric** – metric to compare overlaps (IOS or IOU).
  - **Intersection threshold** – threshold for merging overlaps.

We've empirically determined the optimal combination of parameters for detecting cell nuclei: **NMS** with **IOS** threshold **0.34** (optimal only for NuclePhaser >= 0.4.0).
We recommend changing them only if you have a different task than detecting nuclei.

**Points size** – the size of points in the Points layer (in pixels).

**Bbox thickness** – the thickness of bounding box lines (in pixels).

**Score text size** – the font size of confidence scores (if shown).

**Save result** – if checked, saves the count in a subfolder.

**Save format** – choose CSV, XLSX, or Both.

**Experiment name** – the name of the subfolder for saved results (only used when Save result is enabled). If the folder already exists, a numbered suffix is added.

**Save folder** – the directory where the results will be saved.

Output
++++++

The widget adds one or more layers to the viewer:

- **Points** – a Napari Points layer with a point at the centre of each detection (if **Points** output format is selected).
- **Bounding boxes** – a Napari Shapes layer with rectangles for each detection (if **Bounding boxes** or **Bounding boxes with confidence scores** is selected).
- The number of detected objects is automatically included in the layer name.

Optionally, if **Save result** is enabled, a folder is created with a `.csv` and/or `.xlsx` file containing the count (averaged for TTA) and a metadata file.
