Installation instructions
=========================

Option 1: Using Anaconda (recommended)
++++++++++++++++++++++++++++++++++++++

We recommend installation using `Anaconda Distribution <https://www.anaconda.com/>`_

1. Install Anaconda with `Installation instructions <https://www.anaconda.com/docs/getting-started/anaconda/install>`_

2. Open Anaconda Prompt using Search Bar or Anaconda Navigator

3. Create new environment with default anaconda packages using command

.. code-block:: python

   conda create --name napari-env anaconda

4. Activate new environment using command

.. code-block:: python

   conda activate napari-env

5. Install `Napari <https://napari.org/stable/>`_ using command

.. code-block:: python

   pip install napari[all]

6. Verify napari installation using following command. It should open napari GUI.

.. code-block:: python

   napari

7. Install napari-nuclephaser plugin using command

.. code-block:: python

   pip install napari-nuclephaser

8. Plugin is ready to be used! Start napari by typing

.. code-block:: python

   napari

Initialize plugin's widgets by opening Plugins window and choosing NuclePhaser.

Option 1 advanced: installation with GPU
++++++++++++++++++++++++++++++++++++++++

If you have `NVIDIA GPU with CUDA <https://developer.nvidia.com/cuda-gpus>`_, you can significantly increase plugin's speed.

To install GPU-powered version of the plugin, you first need to do all the steps for the installation using Anaconda (above). Then you need to:

1. Install CUDA using `official instructions <https://developer.nvidia.com/cuda-downloads>`_

.. tip:: Check which versions of CUDA are supported by current `torch installation <https://pytorch.org/get-started/locally/>`_ and consider `installing earlier ones <https://developer.nvidia.com/cuda-toolkit-archive>`_

2. Check CUDA installation with nvidia-smi command in the command line.

.. code-block:: python

   nvidia-smi

3. In the environment with napari and napari-nuclephaser installed, install CUDA-supported torch by typing specific command for your system, which can be found at `torch installation page <https://pytorch.org/get-started/locally/>`_. For example, if you have Windows-based system and CUDA 12.6, your line should look like

.. code-block:: python

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

.. warning:: During our tests, torchvision wasn't installed using this line. To avoid that, add -U after install:
.. code-block:: python

   pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Option 2: Using standalone napari app (simpler)
+++++++++++++++++++++++++++++++++++++++++++++++

.. note:: Downsides of this option are: you can't install GPU-powered version and you will not have CLI (Command Line Interface) that prints detailed progress. Otherwise, it's the same.

1. Download and install napari as standalone app using `installation instructions <https://napari.org/dev/tutorials/fundamentals/installation_bundle_conda.html>`_

2. Search, download and install napari-nuclephaser plugin by opening the app, navigating to Plugins window and choosing Install/Uninstall plugins.
