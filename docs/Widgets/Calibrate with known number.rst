Calibrate with known number
===========================

Description
+++++++++++

This widget is used for finding optimal confidence threshold of the YOLO model for specific use case (cell type, microscopy options etc.)
Learn more about it at :doc:`Confidence threshold calibration page </General information/Confidence threshold calibration>`

This option is the simplest among three, but it doesn't produce accuracy metrics.
The intended task for this widget is calibrating confidence threshold on a **small image**, if there isn't a large one available for calibration.

It takes a calibration image, a number of objects on that image (that was counted beforehand) and a YOLO model and returns an optimal confidence threshold for that model that reutrns closest number of objects to known one.

.. figure:: ../Images/Calibration_known_number.jpg
        :scale: 30 %
        :align: center
        :alt: The image didn't load(

        Graphic representation of how calibrate with known number widget works.

Parameters
++++++++++

**Select image** field is used for selecting an image to run calibration on.
Accepts only single images, if a stack is chosen, an error will be shown.

**Select model** field is used to select YOLO model that will be calibrated.
Currently only small models (n and s) are available due to the limited size of package on PyPI.
We're currently working on adding larger models.

**Calibration number** field is used to enter the known number of objects.
Widget will return the confidence threshold of the YOLO model that will return a number of objects closest to this number.

.. Tip:: You can use uncalibrated model with **Predict on single image widget**, manually correct the result with Napari set of tools and use **Count points on single image** widget to manually count the number of objects on an image.

Further parameters are **advanced settings**. Consider changing them only if you have troubles with default ones.

**SAHI parameters** – these are advanced settings; see :doc:`Sliced inference overview </General information/Sliced inference overview>` for details:
  - **Sahi size** – sliding window size in pixels.
  - **Sahi overlap** – relative overlap between windows.
  - **Postprocess** – algorithm to merge overlapping detections (GREEDYNMM, NMS, NMM).
  - **Match metric** – metric to compare overlaps (IOS or IOU).
  - **Intersection threshold** – threshold for merging overlaps.

We've empirically determined the optimal combination of parameters for detecting cell nuclei: **NMS** with **IOS** threshold **0.34** (optimal only for NuclePhaser >= 0.4.0).
We recommend changing them only if you have a different task than detecting nuclei.

Output
++++++

The widget returns the optimal threshold as a string in the result widget. No files are saved.
