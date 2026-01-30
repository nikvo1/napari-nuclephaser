# NuclePhaser: Cell Proliferation Measurement & Cell Tracking Assistant Plugin for Timelapse Images

[![License MIT](https://img.shields.io/pypi/l/napari-nuclephaser.svg?color=green)](https://github.com/nikvo1/napari-nuclephaser/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nuclephaser.svg?color=green)](https://pypi.org/project/napari-nuclephaser)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nuclephaser.svg?color=green)](https://python.org)
[![tests](https://github.com/nikvo1/napari-nuclephaser/workflows/tests/badge.svg)](https://github.com/nikvo1/napari-nuclephaser/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-nuclephaser)](https://napari-hub.org/plugins/napari-nuclephaser)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A Napari plugin for automated cell nuclei detection, proliferation and population growth analysis, and single-cell tracking in brightfield and fluorescent nuclei timelapse microscopy images.

WARNING!!! Further description isn't displayed correctly at Napari Hub. For correctly displayed description, visit our [GITHUB PAGE](https://github.com/nikvo1/napari-nuclephaser).

napari-nuclephaser is an open-source Napari plugin designed for scientists who need to measure cell proliferation rates, analyze population growth, and perform individual cell tracking on timelapse microscopy images. It utilizes [Ultralytics](https://docs.ultralytics.com/) YOLO object detection models and [obss/sahi](https://github.com/obss/sahi) sliced inference methods to detect cell nuclei on brightfield and fluorescent images of any size, including large whole slide ones. Learn more with [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/index.html) and [paper](https://www.biorxiv.org/content/10.1101/2025.05.13.653705v1).

NEW! If default NuclePhaser models aren't accurate enough on your specific use case, you can finetune them for free with cloud GPU using [Google Colab NuclePhaser finetuning notebook](https://colab.research.google.com/drive/1hKMVQqYS0I_GrkYvdz23tPc8FCv2oJvh?usp=sharing). It is specifically designed for users without coding experience. 

## Nuclei detection

We trained a series of [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv11](https://github.com/ultralytics/ultralytics) models to detect nuclei on phase contrast images. It can be used for counting cells or for individual cell tracking (using nuclei detections as tracking markers). Prominent features of this approach are:
- Napari-nuclephaser plugin includes [obss/sahi](https://github.com/obss/sahi) functionality, allowing detection on images of arbitrary sizes.

<p align="center">
	<picture>
	  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/aa321f17-b0e2-4161-8a69-cb732d7065a7 height=400>
	  <img alt="Image didn't load" src=https://github.com/user-attachments/assets/fe4d6436-3490-4c06-8ddd-7c797976f407 height=400>
	</picture>
</p>

- YOLO models are fast, providing reasonable inference speed even with CPU.
- Ability to predict and automatically count nuclei on stacks of images, making it convenient for cell population growth studies and individual cell tracking.

<p align="center">
	<picture>
	  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/feba9a99-1d37-4962-a2e6-175052aa4925>
	  <img alt="Image didn't load" src="https://github.com/user-attachments/assets/c7e4d0e6-44c1-4268-aae5-6bb78500d928">
	</picture>
 </p>

- Calibration algorithm that allows measuring accuracy for each specific use case.

## Calibration algorithm

Result of object detection model inference is highly dependent on _confidence threshold_ parameter.

<p align="center">
  <picture>
  	<source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/8a13085f-c7ea-45f0-8931-6851f21b68a0 height="300">
  	<img alt="Image didn't load" src=https://github.com/user-attachments/assets/89f76cd7-2db7-4241-bc35-36d23332b2b5 height="300">
  </picture>
</p>

We created several calibration (finding optimal confidence threshold) algorithms that allow adjusting models to specific use cases (cell types, magnifications, illumination settings, cameras etc.):
- Calibration using known number of objects on an image. Doesn't produce accuracy metrics.
- Calibration using fluorescent nuclei stain image (for example, DAPI image). Produces accuracy metrics.
- Calibration using manual annotation of nuclei. Produces accuracy metrics.

Apart from optimal confidence threshold search, these algorithms return accuracy metrics for specific use cases. Given that the calibration image is large, only part of it is used for search of threshold, while the second part is used for evaluation model's accuracy.
Accuracy metrics are [Mean Absolute Percentage Error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) and prediction-ground truth scatterplot, which shows how well model performs with different densities of cells.

NEW! If default NuclePhaser models aren't accurate enough on your specific use case, you can finetune them for free with cloud GPU using [Google Colab NuclePhaser finetuning notebook](https://colab.research.google.com/drive/1hKMVQqYS0I_GrkYvdz23tPc8FCv2oJvh?usp=sharing). It is specifically designed for users without coding experience. 

Learn more about calibration in [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/Biological%20tasks%20guidelines/Individual%20cells%20tracking.html).

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/6d89e22b-2728-40fb-839d-3c6681e29c97>
  <img alt="Image didn't load" src=https://github.com/user-attachments/assets/6a574845-4ad2-4802-b0f8-f1d908aa585a>
  </picture>
</p>

## Cell Proliferation Measurement & Population Growth Analysis

With NuclePhaser you can reconstruct population growth curves from timelapse images of growing cell population by counting number of nuclei on each image. Key features of this approach are:

- No special equipment, reagents or dyes required, only regular culture plastic and cell growth medium, microscope with mechanical stage and a PC (even without GPU).
- [Accuracy measurement for each specific use case](https://napari-nuclephaser.readthedocs.io/en/latest/General%20information/Confidence%20threshold%20calibration.html), so you will be sure the tool is working with appropriate precision.
- Measuring the number of cells, not the area occupied by cells, which can be significantly influenced by spreading/narrowing of cells. 
- Complete reproducibility of results with metadata.txt files saved for each experiment.

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset=https://github.com/user-attachments/assets/47b6cee0-7f4a-440f-84ed-de2a5aa2aa36>
  <img alt="Image didn't load" src=https://github.com/user-attachments/assets/5a084f1a-f977-41fa-b4be-55f37bdf9996>
  </picture>
</p>

For more detailed information about how NuclePhaser can be used for cell proliferation measurement & population growth analysis, visit our [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/Biological%20tasks%20guidelines/Population%20growth%20curves.html#).

## Individual cell tracking

NuclePhaser can be used as an assistant for individual cells tracking. This task is extremely difficult, and manual tracking is still the only method with 100% proof against false tracks. With NuclePhaser, you can significantly simplify manual tracking: instead of marking each cell on each image, you can predict nuclei location with NuclePhaser and then correct the result, which is **much** faster. Learn more in [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/Biological%20tasks%20guidelines/Individual%20cells%20tracking.html).

## Models

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
> Feel free to use the models published here without the plugin!

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

## Citation
If you use NuclePhaser in your work, please cite our preprint:
```bibtex
@article {Voloshin2025.05.13.653705,
	author = {Voloshin, Nikita and Putlyaev, Egor and Chechekhina, Elizaveta and Usachev, Vladimir and Karagyaur, Maxim and Bozov, Kirill and Grigorieva, Olga and Tyurin-Kuzmin, Pyotr and Kulebyakin, Konstantin},
	title = {NuclePhaser: a YOLO-based framework for cell nuclei detection and counting in phase contrast images of arbitrary size with support of fast calibration and testing on specific use cases},
	year = {2025},
	doi = {10.1101/2025.05.13.653705},
	URL = {https://www.biorxiv.org/content/early/2025/05/16/2025.05.13.653705},
	eprint = {https://www.biorxiv.org/content/early/2025/05/16/2025.05.13.653705.full.pdf},
	journal = {bioRxiv}
}
```

## Installation

For detailed installation instructions, visit our [documentation](https://napari-nuclephaser.readthedocs.io/en/latest/Installation/Installation.html).

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

This [napari] plugin was generated with [copier] using the [napari-plugin-template].
