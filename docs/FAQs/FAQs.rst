Frequently asked questions
==========================

Questions about NuclePhaser models
++++++++++++++++++++++++++++++++++

NuclePhaser includes YOLOv5 and YOLOv11 models trained on fluorescent on brightfield images.
Some of them are downloaded automatically, you can download any of them on `NuclePhaser GitHub page <https://github.com/nikvo1/napari-nuclephaser>`_.

- What is the difference between YOLOv5 and YOLOv11?

YOLOv5 and YOLOv11 are different versions of YOLO object detection algorithm developed by `Ultralytics <https://www.ultralytics.com/>`_.
The main difference in general is YOLOv11 are faster and more accurate on `COCO dataset <https://cocodataset.org/#home>`_.
However, that doesn't mean that for specific tasks YOLOv11 will be better.
For example, in NuclePhaser project, the best performing model is in YOLOv5 family.
The other crucial difference: only NuclePhaser YOLOv11 model can be finetuned on custom data using `Google Colab Notebook <https://colab.research.google.com/drive/1hKMVQqYS0I_GrkYvdz23tPc8FCv2oJvh?usp=sharing>`_.

- What is the difference between YOLOv5n, YOLOv5s and so on?

These letters mean nano-small-medium-large-extra large (n-s-m-l-x). In this list, model size increases, which potentially increases accuracy, but slows down inference (application) time.
On validation stage, the best performing model among YOLOv5 and YOLOv11 turned out to be **YOLOv5l**.
However, tests on LIVECell dataset showed that for each cell type a separate model performs best, sometimes even small ones.

- So which model should I choose?

You can start with default models downloaded automatically or with YOLOv5l. If they perform poorly, download and test other models.
There is no guarantee that large models will perform better on your specific conditions, maybe small models will do better. It's all trial and error.

- What if all models show accuracy that is not enough for my project?

You have several options of increasing model's accuracy on your specific data:
    - Finetune YOLOv11 model on your custom data using `Google Colab Notebook <https://colab.research.google.com/drive/1hKMVQqYS0I_GrkYvdz23tPc8FCv2oJvh?usp=sharing>`_. It is specifically designed for users without coding experience.
    - Try :doc:`TTA </General information/Test-time augmentations (TTA)>`.
    - Try :doc:`Dynamic threshold </General information/Dynamic confidence threshold>`.

If you have any other questions, please, don't hesitate to contact us at nikita.voloshin.98@gmail.com.
If your question leads to potential development or error fix in NuclePhaser, you can write a `GitHub issue <https://github.com/nikvo1/napari-nuclephaser/issues>`_.
