# napari-nuclephaser

[![License MIT](https://img.shields.io/pypi/l/napari-nuclephaser.svg?color=green)](https://github.com/nikvo1/napari-nuclephaser/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nuclephaser.svg?color=green)](https://pypi.org/project/napari-nuclephaser)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nuclephaser.svg?color=green)](https://python.org)
[![tests](https://github.com/nikvo1/napari-nuclephaser/workflows/tests/badge.svg)](https://github.com/nikvo1/napari-nuclephaser/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-nuclephaser)](https://napari-hub.org/plugins/napari-nuclephaser)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A Napari plugin to detect and count nuclei on phase contrast images

napari-nuclephaser utilizes [Ultralytics](https://docs.ultralytics.com/) YOLO object detection models and [obss/sahi](https://github.com/obss/sahi) sliced inference methods to detect cell nuclei on phase contrast (and other brightfield) images of any size, including large whole slide ones. Learn more with [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/index.html).

# Nuclei detection

We trained a series of [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv11](https://github.com/ultralytics/ultralytics) models to detect nuclei on phase contrast images. It can be used for counting cells or for individual cell tracking (using nuclei detections as tracking marks). Prominent features of this approach are:
- Napari-nuclephaser plugin includes [obss/sahi](https://github.com/obss/sahi) functionality, allowing detection on images of arbitrary sizes.
  
<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/829d8114-1997-4e28-8c31-e08bfc0c9558 height = "400">
  <source media="(prefers-color-scheme: light)" srcset=https://github.com/user-attachments/assets/d6b7203f-22d2-45f4-8e11-4e2d1691659a height = "400">
  <img alt="Image didn't load" src=https://github.com/user-attachments/assets/829d8114-1997-4e28-8c31-e08bfc0c9558>
  </picture>
</p>

- YOLO models are fast, providing reasonable inference speed even with CPU.
- Ability to predict and automatically count nuclei on stacks of images, making it convenient for cell population growth studies and individual cell tracking.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/feba9a99-1d37-4962-a2e6-175052aa4925>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/c7e4d0e6-44c1-4268-aae5-6bb78500d928">
  <img alt="Image didn't load" src="https://github.com/user-attachments/assets/cf4467ed-9182-4fcb-bd9d-65f853bd5f9b">
</picture>

# Calibration algorithm

Result of object detection model inference is highly dependent on _confidence threshold_ parameter.

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/8a13085f-c7ea-45f0-8931-6851f21b68a0 height = "300">
  <source media="(prefers-color-scheme: light)" srcset=https://github.com/user-attachments/assets/0eb1f7a9-575c-4122-af23-cc8dfead128d height = "300">
  <img alt="Image didn't load" src=https://github.com/user-attachments/assets/c978b11d-611f-41ac-9e5d-f1b8f1649af3>
  </picture>
</p>

We created several calibration (finding optimal confidence threshold) algorithms that allow adjusting models to specific use cases (cell types, magnifications, illumination settings, cameras etc.):
- Calibration using known number of objects on an image. Doesn't produce accuracy metrics.
- Calibration using fluorescent nuclei stain image (for example, DAPI image). Produces accuracy metrics.
- Calibration using manual annotation of nuclei. Produces accuracy metrics.

Apart from optimal confidence threshold search, these algorithms return accuracy metrics for specific use cases. Given that the calibration image is large, only part of it is used for search of threshold, while the second part is used for evaluation model's accuracy.
Accuracy metrics are [Mean Absolute Percentage Error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) and prediction-ground truth scatterplot, which shows how well model performs with different densities of cells.

Learn more about calibration in [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/General%20information/Confidence%20threshold%20calibration.html).

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/6d89e22b-2728-40fb-839d-3c6681e29c97>
  <source media="(prefers-color-scheme: light)" srcset=https://github.com/user-attachments/assets/af53c3ac-0dd4-4a21-8305-aeac4d747839>
  <img alt="Image didn't load">
  </picture>
</p>

# Models

Currently only YOLOv5n, YOLOv5s, YOLOv11n and YOLOv11s models, as well as fluorescent nuclei detector YOLOv5n are downloaded automatically with pip install napari-nuclephaser. To use larger models, download them with these links:

<div align="center">

Fluorescent nuclei detectors
| Model                    | Link |
| :----------------------: | :-----: |
| Fluorescence_v5n         | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v5n.pt?download=1) |
| Fluorescence_v5s         | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v5s.pt?download=1) |
| Fluorescence_v5m         | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v5m.pt?download=1) |
| Fluorescence_v5l         | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v5l.pt?download=1) |
| Fluorescence_v5x         | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v5x.pt?download=1) |
| Fluorescence_v11n        | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v11n.pt?download=1)|
| Fluorescence_v11s        | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v11s.pt?download=1)|
| Fluorescence_v11m        | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v11m.pt?download=1)|
| Fluorescence_v11l        | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v11l.pt?download=1)|
| Fluorescence_v11x        | [Donwload](https://zenodo.org/records/15388030/files/Fluorescence_v11x.pt?download=1)|

Brighfield nuclei detectors
| Model                    | Link |
| :----------------------: | :-----: |
| Brightfield_v5n          | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v5n.pt?download=1)  |
| Brightfield_v5s          | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v5s.pt?download=1)  |
| Brightfield_v5m          | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v5m.pt?download=1)  |
| Brightfield_v5l          | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v5l.pt?download=1)  |
| Brightfield_v5x          | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v5x.pt?download=1)  |
| Brightfield_v11n         | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v11n.pt?download=1) |
| Brightfield_v11s         | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v11s.pt?download=1) |
| Brightfield_v11m         | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v11m.pt?download=1) |
| Brightfield_v11l         | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v11l.pt?download=1) |
| Brightfield_v11x         | [Donwload](https://zenodo.org/records/15388030/files/Brightfield_v11x.pt?download=1) |

</div>

> [!NOTE]
> Feel free to use the models published there without the plugin!

# Plugin functionality
napari-nuclephaser plugin offers following widgets:
- Widget for inference on single image. Result can be in the form of points or boxes with or without confidence scores. Automatically returns number of cells in the name of result layer.
- Widget for inference on stack of images. Optionally can create .csv or .xlsx file at given location with counting results.
- Widget for calibration using known number of cells.
- Widget for calibration using fluorescent nuclei image (fluorescent nuclei detection model is used as a perfect predictor).
- Widget for calibration using manual annotations.
- Widget for transforming Napari Points layer into Labels layer, which allows turning detection in tracking algorithms-digestible form (in particular, [btrack](https://github.com/quantumjot/btrack)).
- Widget for counting number of points in Points layer.

Learn more about widgets and their functionality at [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/index.html).

# Citation

We are currently working on a paper with full description of our approach and how we trained and tested our models.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

# Installation

### Option 1: Using Anaconda (recommended)

We recommend installation using [Anaconda Distribution](https://www.anaconda.com/)

1. Install Anaconda with [Installation instructions](https://www.anaconda.com/docs/getting-started/anaconda/install)

2. Open Anaconda Prompt using Search Bar or Anaconda Navigator

3. Create new environment with default anaconda packages using command

```sh
conda create --name napari-env anaconda
```
4. Activate new environment using command

```sh
conda activate napari-env
```
5. Install [Napari](https://napari.org/stable/) using command

```sh
pip install napari[all]
```
6. Verify napari installation using following command. It should open napari GUI.

```sh
napari
```
7. Install napari-nuclephaser plugin using command

```sh
pip install napari-nuclephaser
```
8. Plugin is ready to be used! Start napari by typing

```sh
napari
```
Initialize plugin's widgets by opening Plugins window and choosing NuclePhaser.

### Installation with GPU

If you have [NVIDIA GPU with CUDA](https://developer.nvidia.com/cuda-gpus), you can significantly increase plugin's speed.

To install GPU-powered version of the plugin, you first need to do all the steps for the installation using Anaconda (above). Then you need to:

1. Install CUDA using [official instructions](https://developer.nvidia.com/cuda-downloads)
> [!NOTE]
> Check which versions of CUDA are supported by current [torch installation](https://pytorch.org/get-started/locally/) and consider [installing earlier ones](https://developer.nvidia.com/cuda-toolkit-archive)

2. Check CUDA installation with nvidia-smi command in the command line.

```sh
nvidia-smi
```
3. In the environment with napari and napari-nuclephaser installed, install CUDA-supported torch by typing specific command for your system, which can be found at [torch installation page](https://pytorch.org/get-started/locally/). For example, if you have Windows-based system and CUDA 12.6, your line should look like

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> [!WARNING]
> During our tests, torchvision wasn't installed using this line. To avoid that, add -U after install:
> ```sh
>pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
>```

----------------------------------

### Option 2: Using standalone napari app (simpler)

1. Download and install napari as standalone app using [installation instructions](https://napari.org/dev/tutorials/fundamentals/installation_bundle_conda.html)

2. Search, download and install napari-nuclephaser plugin by opening the app, navigating to Plugins window and choosing Install/Uninstall plugins...

----------------------------------

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-nuclephaser" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/nikvo1/napari-nuclephaser/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
