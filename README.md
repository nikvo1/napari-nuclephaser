# napari-nuclephaser

[![License MIT](https://img.shields.io/pypi/l/napari-nuclephaser.svg?color=green)](https://github.com/nikvo1/napari-nuclephaser/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nuclephaser.svg?color=green)](https://pypi.org/project/napari-nuclephaser)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nuclephaser.svg?color=green)](https://python.org)
[![tests](https://github.com/nikvo1/napari-nuclephaser/workflows/tests/badge.svg)](https://github.com/nikvo1/napari-nuclephaser/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-nuclephaser)](https://napari-hub.org/plugins/napari-nuclephaser)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A Napari plugin to detect and count nuclei on phase contrast images

napari-nuclephaser utilizes [Ultralytics](https://docs.ultralytics.com/) YOLO object detection models and [obss/sahi](https://github.com/obss/sahi) sliced inference methods to detect cell nuclei on phase contrast (and other brightfield) images of any size, including large whole slide ones. 

# Nuclei detection

We trained a series of YOLOv5 and YOLOv11 models to detect nuclei on phase contrast images. It can be used for counting cells or for individual cell tracking (using nuclei detections as tracking marks). Prominent features of this approach are:
- Napari-nuclephaser plugin inclues [obss/sahi](https://github.com/obss/sahi) functionality, allowing detection on images of arbitrary sizes. 
- YOLO models are fast, providing reasonable inference speed even with CPU.
- Ability to predict and automatically count nuclei on stacks of images, making it convenient for cell population growth studies and individual cell tracking.

# Calibration algorithm

Result of object detection model inference is highly dependent on _confidence threshold_ parameter.


----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-nuclephaser` via [pip]:

    pip install napari-nuclephaser



To install latest development version :

    pip install git+https://github.com/nikvo1/napari-nuclephaser.git


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
