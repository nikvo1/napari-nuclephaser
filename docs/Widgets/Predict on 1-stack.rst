Predict on 1-stack
==================

Description
+++++++++++

This widget runs :doc:`YOLO </General information/Object detection overview>` with :doc:`SAHI </General information/Sliced inference overview>` on a **one‑dimensional stack** of images. Images can be any size and format.

The widget can run in three **Detection modes**:
    - **Regular detection** – uses a fixed confidence threshold.
    - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` – uses test‑time augmentations (requires a metadata file from :doc:`Calibrate with points </Widgets/Calibrate with points>`).
    - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` – uses a dynamic threshold model (requires a `.pkl` file from :doc:`Calibrate with dynamic threshold </Widgets/Calibrate dynamic threshold>`).

The widget adds a **Points layer** for each frame (or a combined layer) containing the filtered detections. It can also save a count table (CSV/XLSX) for each frame.

Parameters
++++++++++

**Select stack** – the 1‑dimensional stack of images. Only 1‑stacks are accepted; single images or 2‑stacks will raise an error.

**Select model** – the YOLO model to use.

**Detection mode** – choose between:
  - Regular detection
  - Detection with :doc:`TTA </General information/Test-time augmentations (TTA)>` (requires a `.txt` metadata file)
  - Detection with :doc:`Dynamic threshold </General information/Dynamic confidence threshold>` (requires a `.pkl` model file)

**Mode file** – the file needed for TTA or dynamic threshold modes (`.txt` for TTA, `.pkl` for dynamic threshold). This field is only active when the corresponding mode is selected.

**Confidence threshold** – used in **Regular detection** mode.

**SAHI parameters** – these are advanced settings; see :doc:`Sliced inference overview </General information/Sliced inference overview>` for details:
  - **Sahi size** – sliding window size in pixels.
  - **Sahi overlap** – relative overlap between windows.
  - **Postprocess** – algorithm to merge overlapping detections (GREEDYNMM, NMS, NMM).
  - **Match metric** – metric to compare overlaps (IOS or IOU).
  - **Intersection threshold** – threshold for merging overlaps.

We've empirically determined the optimal combination of parameters for detecting cell nuclei: **NMS** with **IOS** threshold **0.34** (optimal only for NuclePhaser >= 0.4.0).
We recommend changing them only if you have a different task than detecting nuclei.

**Points size** – the size of points in the output Points layer.

**Save result** – if checked, saves per‑frame counts.

**Save format** – CSV, XLSX, or Both.

**Experiment name** – subfolder name.

**Save folder** – directory for results.

Output
++++++

- A **Points layer** (3‑dimensional: frame, y, x) containing the detections for the whole stack.
- If **Save result** is enabled, a subfolder is created with a `.csv` and/or `.xlsx` file containing the per‑frame counts and a metadata file.
