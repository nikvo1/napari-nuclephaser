Predict on 2-stack
==================

Description
+++++++++++

This widget runs :doc:`YOLO </General information/Object detection overview>` with :doc:`SAHI </General information/Sliced inference overview>` on a **two‑dimensional stack** of images (e.g., multi‑well plate or time‑lapse with multiple conditions). Images can be any size and format.

The widget can run in three **Detection modes**:
    - **Regular detection** – uses a fixed confidence threshold.
    - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` – uses test‑time augmentations (requires a metadata file from :doc:`Calibrate with points </Widgets/Calibrate with points>`).
    - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` – uses a dynamic threshold model (requires a `.pkl` file from :doc:`Calibrate with dynamic threshold </Widgets/Calibrate dynamic threshold>`).

The widget adds a **Points layer** containing the detections for the whole stack. It can save a count table (CSV/XLSX) for each (dim1, dim2) pair.

Parameters
++++++++++

**Select stack** – the 2‑dimensional stack. Only 2‑stacks are accepted.

**Select model** – the YOLO model.

**Detection mode** – choose between:
  - Regular detection
  - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` (requires a `.txt` metadata file)
  - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` (requires a `.pkl` model file)

**Mode file** – the file needed for TTA or dynamic threshold modes (`.txt` for TTA, `.pkl` for dynamic threshold). This field is only active when the corresponding mode is selected.

**Confidence threshold** – used in Regular detection.

**SAHI parameters** – these are advanced settings; see :doc:`Sliced inference overview </General information/Sliced inference overview>` for details:
  - **Sahi size** – sliding window size in pixels.
  - **Sahi overlap** – relative overlap between windows.
  - **Postprocess** – algorithm to merge overlapping detections (GREEDYNMM, NMS, NMM).
  - **Match metric** – metric to compare overlaps (IOS or IOU).
  - **Intersection threshold** – threshold for merging overlaps.

We've empirically determined the optimal combination of parameters for detecting cell nuclei: **NMS** with **IOS** threshold **0.34** (optimal only for NuclePhaser >= 0.4.0).
We recommend changing them only if you have a different task than detecting nuclei.

**Points size** – point size in the output layer.

**Save result** – if checked, saves counts.

**Save format** – CSV, XLSX, or Both.

**Experiment name** – subfolder name.

**Save folder** – directory for results.

Output
++++++

- A **Points layer** with detections.
- If **Save result** is enabled, a subfolder is created with a `.csv` and/or `.xlsx` file containing counts for each (dim1, dim2) pair and a metadata file.
