Calibrate with points
=====================

Description
+++++++++++

This widget is used for finding optimal confidence threshold of the YOLO model for specific use case (cell type, microscopy options etc.)
Learn more about it at :doc:`Confidence threshold calibration page </General information/Confidence threshold calibration>`.

.. note::
        In NuclePhaser >= 0.2.5, this widget supports simultaneous calibration on multiple images! Use 1-dimensional stack of images and points.

.. figure:: ../Images/Calibration_and_test.jpg
        :scale: 20 %
        :align: center
        :alt: The image didn't load(

        Workflow diagram of Calibrate with DAPI widget. Calibrate with points widget works the same way, but user's manual annotations are used as ground truth instead of fluorescent nuclei.

.. note:: You need a large image(s) for the use of that widget, the larger - the better. At least 6400x6400 pixels is recommended.

You need a large image(s) for that option and Napari Points layer with marked nuclei for that image(s).
You have three options of creating that layer:

.. hint:: Second and third options are much faster!

* Manually label all the nuclei. Above the image layer icon on the left, press the New point layer button (the left one with six dots). Use `Napari set of tools <https://napari.org/dev/howtos/layers/points.html>`_ to label all the nuclei.
* Manually correct annotations of uncalibrated model. Use :doc:`Predict on single image widget </Widgets/Predict on single image>` or :doc:`Predict on 1-stack </Widgets/Predict on 1-stack>` with arbitrary confidence threshold and correct the result Points layer using `Napari set of tools <https://napari.org/dev/howtos/layers/points.html>`_. In our practice, adding missing points is more convenient than deleting extra, so we use higher confidence threshold.
* (Optimal) Manually correct annotations of fluorescent nuclei detector on images with fluorescent nuclei stain. Use :doc:`Predict on single image widget </Widgets/Predict on single image>` or :doc:`Predict on 1-stack </Widgets/Predict on 1-stack>` with arbitrary confidence threshold and correct the result Points layer using `Napari set of tools <https://napari.org/dev/howtos/layers/points.html>`_. Downside: it requires staining of samples.

.. figure:: ../Images/Napari_tools.jpg
        :scale: 40 %
        :align: center
        :alt: The image didn't load(

        Napari set of tools to edit Points layer. Circle with plus sign inside (Second tool) is used for adding new points. Use arrow (Third tool) to select extra points (with pressed Ctrl to select several) and Delete button or Cross (First tool) to remove extra markers.


Parameters
++++++++++

**Select Phase image** field is used for selecting the brightfield image(s) that will be used for calibration.

**Select points layer** field is used for selecting Napari Points layer with manual annotations of nuclei (see description above for options of creating it)

**Phase model** is used for selecting model that will be calibrated.

**Division size** determines the amount of small images that your large images will be split into.
It defines the size of one small image in pixels.
For example, if you have an image 6400x6400 pixels, and Division size = 640, your result array will contain a 100 small images.

**Calibration proportion** determines which part of small images array will be used for calibration, and which part - for test.
If you have an array of 100 small images and Calibration proportion = 0,1, 10 of those images will be used for calibration, 90 - for test.

**Test with TTA** checkbox is used for running calibration and test with TTA (test-time augmentations).
Learn more at :doc:`page about TTA </General information/Test-time augmentations (TTA)>`.

**Save folder** is used for selecting a folder in which the calibration plot and metadata.txt files will be saved.
Inside this folder, a subfolder will be created with Expereiment name.

**Experiment name** is used for setting up the subfolder name in **Save folder** for saving the results.
If such folder already exists, will create another subfolder with *Experiment name1* or *Experiment name2* etc.

Further parameters are **advanced settings**. Consider changing them only if you have troubles with default ones.

**Random seed** is used for exact reproduction of data.
The calibration and test parts are divided randomly, using the same random seed will result in the same division.

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

The widget creates a folder with:

- **Calibration_error_plot.png** – a scatterplot of ground truth counts (from points) vs. predicted counts, with MAPE shown.
- **reference_points.csv** – the points used for calibration (for reproducibility).
- **metadata.txt** – all parameters, per‑frame thresholds, and MAPE details.
- If TTA is enabled, a **TTA** subfolder with the best combination and its MAPE is also created.
