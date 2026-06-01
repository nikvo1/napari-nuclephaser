Test-time augmentations (TTA)
=============================

Make sure you are familiar with :doc:`Confidence threshold calibration </General information/Confidence threshold calibration>`.

What are test-time augmentations (TTA)?
Put simply, they are a way to potentially increase accuracy by sacrificing inference time.

Imagine you have a calibration image, and the model has an error of 10% on it.
But what if, instead of using this native image, it would be slightly changed: contrasted, magnified or something else?
It is possible that model would be better at these augmented images.
For example, cells on calibration image are small, and resizing the image 1.5x resulted in error of 8%.
Or even better, when we average the results from native and magnified image, their respective errors cancel each other, and the result error can be even lower!

This is exactly what TTA are about.
On calibration stage, instead of running single calibration on native image, it gets augmented in the following ways:

**resize_1.5x** - image is increased in size 1.5x

**resize_2x** - image is increased in size 2x

**clahe** - `CLAHE (contrast-limited adaptive histogram equalization) <https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE>`_ is applied to the image.

**gamma 1.5** - `gamma contrast <https://en.wikipedia.org/wiki/Gamma_correction>`_ with coefficient 1.5 is applied to the image

**invert** - all image intensities are inverted (255 becomes 0, 0 becomes 255 and so on)

**median_3** - `median filter <https://en.wikipedia.org/wiki/Median_filter>`_ with kernel size 3 is applied to the image

**bilateral_10** - `bilateral filter <https://en.wikipedia.org/wiki/Bilateral_filter>`_ with sigma value 10 is applied to the image

**sharpen** - `unsharp mask <https://en.wikipedia.org/wiki/Unsharp_masking>`_ is applied to the image.

Best threshold is found for each augmentation independently.
The image is used for calibration with each augmentation applied, and then it gets tested on each possible combination of these augmentations.

.. note:: How combination of augmentations work? Each augmentation is applied individually and results from them are averaged, not all of augmentations are applied simultaneously.

The main tradeoff is inference time. For example, the combination of 5 augmentations decreases the error from 10% to 9%.
It means that for increasing accuracy by 1% you will have to spend 5x more time on inference.

You can calibrate with TTA using :doc:`Calibrate with points widget </Widgets/Calibrate with points>`.
The widget creates a "TTA" subfolder in the given folder for storing the calibration results.
metadata_TTA.txt file is stored in that subfolder.
All inference (predict on single image, predict on 1-stack, predict on 2-stack) widgets have "Use TTA" checkbox and a field for selecting .txt file.
Select your metadata_TTA.txt calibration file, and widget will perform TTA automatically.
Widget will create a points layer for each augmentation, and the result .csv or .xlsx file will contain averaged counting results.
