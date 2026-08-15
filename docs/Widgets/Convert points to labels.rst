Convert points to labels
========================

Description
+++++++++++

This widget is used in :doc:`Individual cells tracking pipeline </Biological tasks guidelines/Individual cells tracking>`.

While :doc:`Predict on 1-stack widget </Widgets/Predict on 1-stack>` returns detections in form of Napari.Points layer, tracking algorithms demand markers in a different format.
In particular, `btrack <https://github.com/quantumjot/btrack>`_ plugin needs Napari.Labels layer.

This widget converts Napari.Points layer into btrack-digestible Napari.Labels layer.

Parameters
++++++++++

**Select points layer** field is used for selecting the Napari.Points layer that you want to convert into labels.

**Select reference image** field is used for selecting the stack of images that you want to perform tracking on.
Behind the scenes, points in Napari.Points layer are defined relatively, so reference image is required for mapping the points.

**Label size** is used for determining the size of label that each point will be converted into.
At each point, a circle with this size will be created.

.. warning:: You should be careful with Label size! If it will be large, two points located nearby can merge into one label!

.. hint:: You can use Label size to your advantage! If you have a lot of false positives that are located very closely to true positives, you can set Label size large enough for them to merge!

Output
++++++

Adds a Napari Labels layer with circles at each point position, with a diameter equal to **Label size**.
