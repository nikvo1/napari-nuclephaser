Calibrate with DAPI
===================

Description
+++++++++++

This widget is used for finding optimal confidence threshold of the YOLO model for specific use case (cell type, microscopy options etc.)
Learn more about it at :doc:`Confidence threshold calibration page </General information/Confidence threshold calibration>`.

.. warning::
        This widget is suboptimal for calibration - if the fluorescent detector makes mistakes, there is no way to correct them with that widget.
        :doc:`Calibrate with points widget </Widgets/Calibrate with points>` is more reliable: you can run fluorescent model with :doc:`Predict on single image </Widgets/Predict on single image>` or :doc:`Predict on 1-stack </Widgets/Predict on 1-stack>`, manually correct detections and then calibrate.

        .. note::
        This widget doesn't support simultaneous calibration on multiple images, while :doc:`Calibrate with points widget </Widgets/Calibrate with points>` does!

.. figure:: ../Images/Calibration_pair.jpg
        :scale: 30 %
        :align: center
        :alt: The image didn't load(

        An example pair of images you need to have for this calibration method.

.. note:: You need a large image for the use of that widget, the larger - the better. At least 6400x6400 pixels is recommended.

Behind the scenes it works the following way. At first, large images (brightfield and fluorescence) are split into an array of small ones.
Part of this array is used for calibration: for each pair, ground truth is derived from fluorescence image and fluorescent nuclei detector, which is used as a "perfect predictor".
For the calibrated model, the confidence threshold returning the closest number of objects to the "perfect predictor" is found.
The result calibrated threshold is calculated as the mean between all calibration small images.

.. figure:: ../Images/Calibration_and_test.jpg
        :scale: 20 %
        :align: center
        :alt: The image didn't load(

        Workflow diagram of Calibrate with DAPI widget.

After calibration, the test algorithm will run. For the test part of small images array, a calibrated model and "perfect predictor" are applied and counting results are compared.
Then two metrics are calculated: `MAPE <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_ and prediction-ground truth scatterplot.
The less the MAPE, the better. In scatterplot, each point represents an image. The closer points are to the red line (imaginary line of perfect predictions), the better.

The widget also saves metadata.txt file with detailed information about calibration run for the reproducibility of results.

Parameters
++++++++++

**Select Phase image** field is used for selecting the brightfield image of your pair.

**Select DAPI image** field is used for selecting the fluorescence image of your pair.

**Phase model** is used for selecting model that will be calibrated. Models can be downloaded on `NuclePhaser GitHub page <https://github.com/nikvo1/napari-nuclephaser>`_.

**DAPI model** is used for selecting the fluorescence nuclei detector that will be used as a "perfect predictor". Models can be downloaded on `NuclePhaser GitHub page <https://github.com/nikvo1/napari-nuclephaser>`_.

.. hint:: Use **Predict on single image** widget beforehand to test and experiment with how this "perfect predictor" you want to use performs.

**Division size** determines the amount of small images that your large images will be split into.
It defines the size of one small image in pixels.
For example, if you have an image 6400x6400 pixels, and Division size = 640, your result array will contain a 100 small images.

**Calibration proportion** determines which part of small images array will be used for calibration, and which part - for test.
If you have an array of 100 small images and Calibration proportion = 0,1, 10 of those images will be used for calibration, 90 - for test.

**DAPI confidence threshold** is used for setting up the confidence threshold of "perfect predictor" model.

.. hint:: Use **Predict on single image** widget beforehand to test and experiment with how this "perfect predictor" you want to use performs.

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

The widget creates a subfolder with:

- **Calibration error plot.png** – a scatterplot comparing DAPI‑based ground truth counts (x‑axis) versus model predictions (y‑axis), with MAPE shown in the title.
- **metadata.txt** – a complete record of all parameters, thresholds, and MAPE values.
