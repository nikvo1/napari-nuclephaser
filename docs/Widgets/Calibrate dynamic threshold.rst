Calibrate with dynamic threshold
================================

Description
+++++++++++

This widget is used for training a **dynamic threshold** for the YOLO model on a specific use case (cell type, microscopy options etc.).

Learn more about the concept at the :doc:`Dynamic confidence threshold page </General information/Dynamic confidence threshold>`.

While the standard :doc:`Calibrate with points </Widgets/Calibrate with points>` method finds a **single global threshold** for the image/stack of images, the **dynamic threshold** adapts the threshold **locally** – different regions of the image receive different thresholds based on their local properties.

This is particularly useful when **conditions vary** within a single image or across frames:

- Different contrast.
- Regions with high and low cell density.
- Local defocus.

The widget trains a model that learns the relationship between **local image features** and the **optimal local threshold** from your manual annotations. The trained model is saved as a ``.pkl`` file and can be used in any inference widget for local threshold adaptation.

Parameters
++++++++++

**Select Phase image** field is used for selecting the brightfield image (or stack) that will be used for calibration. Accepts a single image or a 1‑dimensional stack.

**Select points layer** field is used for selecting the Napari Points layer with manual annotations of nuclei. The points should correspond to the **Select Phase image**. You can create this layer manually or by correcting the result of an uncalibrated model (see :doc:`Calibrate with points </Widgets/Calibrate with points>` for details).

**Phase model** is used for selecting the YOLO model that will be calibrated.

**Division size** determines the size of the small patches that the image will be split into during calibration. Each patch is processed independently.

**Calibration proportion** determines which fraction of patches will be used for calibration and which for testing. For example, 0.5 means half the patches are used for training the model, and the other half for evaluating its performance.

**Max blur strength** determines the maximum defocus level the model will adapt to. The algorithm tests Gaussian blur sigmas from 0 (no blur) up to this value. Higher values make the model more robust to defocus but increase calibration time. Set to 0 to skip defocus adaptation.

**Save folder** is used for selecting the folder in which the calibration results will be saved.

**Experiment name** is used for setting the subfolder name in **Save folder**. If such a folder already exists, a new one with a number suffix (e.g., ``Experiment_name1``) will be created.

**Advanced settings** – consider changing parameters below only if you have trouble with the default values.

**Random seed** ensures exact reproducibility of the calibration/test split.

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

The widget produces the following outputs in the selected **Save folder**:

- **dynamic_threshold.pkl** – the trained dynamic threshold model file. You will need this file for inference with dynamic threshold.

- **reference_points.csv** – the reference points used for calibration (for reproducibility).

- **Error_plot_for_blur_sigma_X.png** – scatterplots comparing dynamic vs static threshold performance for each blur strength tested. The closer the points to the red line, the better.

- **metadata.txt** – a text file containing all parameters used for the calibration run, the training details (number of samples, features used, best k), the static and dynamic MAPE per blur strength, and a comparison summary.
