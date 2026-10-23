# Poker card recognition detection system based on YOLOv11 in Python source code + ONNX model + evaluation metrics curve + exquisite GUI interface

[Algorithm Introduction]The poker card recognition detection system based on YOLOv11 is a system that utilizes deep learning technology for the identification of poker card types and card face information.

[Algorithm Introduction]

A poker card recognition detection system based on YOLOv11 is a system that utilizes deep learning technology for the identification and recognition of poker card types and face information. This system employs YOLOv11 (You Only Look Once version 11) as the core algorithm, which is an efficient target detection algorithm capable of transforming the target detection task into a regression problem, directly outputting the category and location of the target.

YOLOv11 is the latest version of the YOLO series, which has improved detection accuracy, speed, and applicability compared to previous versions. This makes the poker card recognition detection system based on YOLOv11 more accurate and faster in identifying poker cards.

This system can be applied in various fields, such as intelligent card games, online poker cheat detection, and teaching aids. In the context of intelligent card games, the system can automatically identify the cards held by players, enabling automatic scoring and determining the winner, thereby enhancing the fun and efficiency of the game. In terms of online poker cheat detection, the system can monitor the game in real-time to detect any cheating activities, ensuring the fairness and impartiality of the game.

Furthermore, Poker card identification detection system can also be used for the quality inspection of poker cards. Automated packaging and sorting Special places such as safety monitoring and other scenarios. For example, In the production process of playing cards, The system can automatically inspect the quality of the produced playing cards. Ensure product quality meets standards.

In general, the poker card recognition detection system based on YOLOv11 has broad application prospects and important practical value.

[Effect Display]

![](01-9d2f32b4.png)

![](02-90c632cd.png) 

[Test Environment]

windows10
anaconda3+python3.8
torch==2.3.0
ultralytics==8.3.81
onnxruntime==1.16.3

The model can detect 52 categories.

10C,10D,3H,3S,4C,4D,4H,4S,5C,5D,5H,5S,10H,6C,6D,6H,6S,7C,7D,7H,7S,8C,8D,10S,8H,8S,9C,9D,9H,9S,AC,AD,AH,AS,2C,JC,JD,JH,JS,KC,KD,KH,KS,QC,QD,2D,QH,QS,2H,2S,3C,3D

[Training Dataset Introduction]

Dataset format: Pascal VOC format + YOLO format (txt file without split path, only containing jpg images and corresponding VOC format xml files and yolo format txt files)
Number of images (number of jpg files): 24233
Number of annotations (number of xml files): 24233
Number of annotations (number of txt files): 24233
Number of annotation categories: 52
Label category names (Note that the order in yolo format does not correspond to this, but is based on the classes.txt in the labels folder): ["2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S", "9C", "9D", "9H", "9S", "10C", "10D", "10H", "10S", "AC", "AD"]"AH","AS","JC","JD","JH","JS","KC","KD","KH","KS","QC","QD","QH","QS"]
Number of boxes labeled in each category:
2C frame count = 1835
2D Frame Count = 1829
2H Frame Count = 1784
2S frame count = 1843
3C Frame Count = 1946
3D Frames = 1747
3H Frame Count = 1804
3S frame count = 1860
4C frame count = 1983
4D frame count = 1977
4H frame count = 1957
4S frame count = 1767
5C Frames = 2020
5D Frame Count = 1934
5H frame count = 1737
5S box count = 1910
6C frame count = 1768
6D Frame Count = 1850
6H frame count = 1743
6S box count = 1876
7C Frames = 1796
7D frame count = 1813
7H Frames = 1911
7S frame count = 1932
8C Frames = 1786
8D Frame Count = 1895
8H Frame Count = 1892
8S frame count = 1788
9C Frames = 1774
9D Frame Count = 1910
9H Frames = 1893
9S frame count = 1775
10C frame count = 1847
10D Frames = 1810
10H Frame Count = 1897
10S frame count = 1937
AC Frames = 1903
AD Frames = 1934
AH Frame Count = 1805
AS Frames = 1908
JC Frames = 1965
JD Frames = 1706
JH Frames = 1720
JS Frame Count = 1810
KC Frames = 1892
KD Frames = 1916
KH Frame Count = 1876
KS Frames = 1902
QC Frames = 1906
QD Frame Count = 1940
QH Frame Count = 1891
QS Frame Count = 2009
Total frame count: 96909
Use LabelImg: labelImg
Annotation rules: draw a rectangle around the category.
Important note: The dataset includes enhanced images.
Special statement: This dataset does not guarantee the accuracy or weights of the trained model or weight files, only accurate and reasonable annotations are provided.

Image Preview:

![](03-56c6e2df.png)

Annotation example:

![](04-34ac40fa.jpg)

[Training Information]

|

Parameters

 ||

Value

 |
|

Number of training set images

 ||

16963

 |
|

Number of images in the validation set

 ||

4847

 |
|

Training map

 ||

99.5%

 |
|

Training Accuracy

 ||

99.9%

 |
|

Training recall rate

 ||

99.9%

 |

Validation set test accuracy information

|

Category

 ||

MAP50

 |
|

all

 ||

99

 |
|

10C

 ||

100

 |
|

10D

 ||

100

 |
|

3H

 ||

100

 |
|

3S

 ||

100

 |
|

4C

 ||

100

 |
|

4D

 ||

100

 |
|

4H

 ||

100

 |
|

4S

 ||

100

 |
|

5C

 ||

100

 |
|

5D

 ||

100

 |
|

5H

 ||

100

 |
|

5S

 ||

100

 |
|

10H

 ||

100

 |
|

6C

 ||

100

 |
|

6D

 ||

100

 |
|

6H

 ||

100

 |
|

6S

 ||

100

 |
|

7C

 ||

100

 |
|

7D

 ||

100

 |
|

7H

 ||

100

 |
|

7S

 ||

100

 |
|

8C

 ||

100

 |
|

8D

 ||

100

 |
|

10S

 ||

100

 |
|

8H

 ||

100

 |
|

8S

 ||

100

 |
|

9C

 ||

100

 |
|

9D

 ||

100

 |
|

9H

 ||

100

 |
|

9S

 ||

100

 |
|

AC

 ||

100

 |
|

AD

 ||

100

 |
|

AH

 ||

100

 |
|

AS

 ||

100

 |
|

2C

 ||

99

 |
|

JC

 ||

100

 |
|

JD

 ||

100

 |
|

JH

 ||

100

 |
|

JS

 ||

100

 |
|

KC

 ||

100

 |
|

KD

 ||

100

 |
|

KH

 ||

100

 |
|

KS

 ||

100

 |
|

QC

 ||

99

 |
|

QD

 ||

100

 |
|

2D

 ||

100

 |
|

QH

 ||

100

 |
|

QS

 ||

100

 |
|

2H

 ||

99

 |
|

2S

 ||

100

 |
|

3C

 ||

99

 |
|

3D

 ||

100

 |

[Interface Design]

-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-

```
import osimport sysfrom PyQt5 import QtCore, QtGui, QtWidgetsfrom PyQt5.QtCore import QThread, pyqtSignalfrom PyQt5.QtGui import QImage, QPixmapfrom PyQt5.QtWidgets import QFileDialog, QLabel, QApplicationimport image_rcimport threadingimport cv2import numpy as npimport timefrom Yolo11Detector import *
class Ui_MainWindow(QtWidgets.QMainWindow):    signal = QtCore.pyqtSignal(str, str)
    def setupUi(self):        self.setObjectName("MainWindow")        self.resize(1280, 728)        self.centralwidget = QtWidgets.QWidget(self)        self.centralwidget.setObjectName("centralwidget")        self.detector=None        self.weights_dir = './weights'
        self.picture = QtWidgets.QLabel(self.centralwidget)        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))        self.picture.setStyleSheet("background:black")        self.picture.setObjectName("picture")        self.picture.setScaledContents(True)        self.label_2 = QtWidgets.QLabel(self.centralwidget)        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))        self.label_2.setObjectName("label_2")        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))        self.cb_weights.setObjectName("cb_weights")        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))        self.label_3.setObjectName("label_3")        self.hs_conf = QtWidgets.QSlider(self.centralwidget)        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))        self.hs_conf.setProperty("value", 25)        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)        self.hs_conf.setObjectName("hs_conf")        self.hs_conf.valueChanged.connect(self.conf_change)        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))        self.dsb_conf.setMaximum(1.0)        self.dsb_conf.setSingleStep(0.01)        self.dsb_conf.setProperty("value", 0.3)        self.dsb_conf.setObjectName("dsb_conf")        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))        self.dsb_iou.setMaximum(1.0)        self.dsb_iou.setSingleStep(0.01)        self.dsb_iou.setProperty("value", 0.45)        self.dsb_iou.setObjectName("dsb_iou")        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)        self.hs_iou = QtWidgets.QSlider(self.centralwidget)        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))        self.hs_iou.setProperty("value", 45)        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)        self.hs_iou.setObjectName("hs_iou")        self.hs_iou.valueChanged.connect(self.iou_change)        self.label_4 = QtWidgets.QLabel(self.centralwidget)        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))        self.label_4.setObjectName("label_4")        self.label_5 = QtWidgets.QLabel(self.centralwidget)        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))        self.label_5.setObjectName("label_5")        self.le_res = QtWidgets.QTextEdit(self.centralwidget)        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))        self.le_res.setObjectName("le_res")        self.setCentralWidget(self.centralwidget)        self.menubar = QtWidgets.QMenuBar(self)        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))        self.menubar.setObjectName("menubar")        self.setMenuBar(self.menubar)        self.statusbar = QtWidgets.QStatusBar(self)        self.statusbar.setObjectName("statusbar")        self.setStatusBar(self.statusbar)        self.toolBar = QtWidgets.QToolBar(self)        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)        self.toolBar.setObjectName("toolBar")        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)        self.actionopenpic = QtWidgets.QAction(self)        icon = QtGui.QIcon()        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.actionopenpic.setIcon(icon)        self.actionopenpic.setObjectName("actionopenpic")        self.actionopenpic.triggered.connect(self.open_image)        self.action = QtWidgets.QAction(self)        icon1 = QtGui.QIcon()        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.action.setIcon(icon1)        self.action.setObjectName("action")        self.action.triggered.connect(self.open_video)        self.action_2 = QtWidgets.QAction(self)        icon2 = QtGui.QIcon()        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.action_2.setIcon(icon2)        self.action_2.setObjectName("action_2")        self.action_2.triggered.connect(self.open_camera)
        self.actionexit = QtWidgets.QAction(self)        icon3 = QtGui.QIcon()        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.actionexit.setIcon(icon3)        self.actionexit.setObjectName("actionexit")        self.actionexit.triggered.connect(self.exit)
        self.toolBar.addAction(self.actionopenpic)        self.toolBar.addAction(self.action)        self.toolBar.addAction(self.action_2)        self.toolBar.addAction(self.actionexit)
        self.retranslateUi()        QtCore.QMetaObject.connectSlotsByName(self)        self.init_all()
```

[Training Steps]

To train your own dataset using YOLO11, you need to follow some basic steps. YOLO11 is a version of the YOLO series model that has made many improvements over its predecessors, including more efficient training processes and higher accuracy. Here are the detailed steps for training your own YOLO format dataset:

1. Preparation of the environment

1. Install necessary software: Ensure that Python (recommended version 3.6 or higher) is installed on your computer, as well as CUDA and cuDNN (if you plan to use a GPU for acceleration).

2. Install YOLOv8 library: You can clone the YOLOv8 repository on GitHub or directly install YOLO11 via pip. For example:
    pip install ultralytics

II. Data Preparation

3. Organize data structure: Organize your data folder according to the requirements of YOLO. Typically, you need a directory structure that includes image and label files, such as:

   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/

Among them, train and val represent the training set and validation set respectively. Moreover, the names of the images folder and labels folder cannot be arbitrarily modified or written incorrectly, otherwise, they will not be able to find the dataset during the training process.

4. Annotation Data: Use appropriate tools to label the images, generating YOLO formatted label files. Each label file should be a .txt file, with each line representing a bounding box in the format:

<Category ID> <Center Point X> <Center Point Y> <Width> <Height>

These values are normalized relative to the image size.

5. Create data configuration file: Create a .yaml file to define your dataset, including path, category list, etc. For example:
   # dataset.yaml
Path: ./dataset
train: images/train # Training image relative path
val: images/val # Verify the relative path of the validation image
nc: 2 # Number of categories
Names: ['class1', 'class2'] # Category names

III. Model Training

6. Load pre-trained models: Use the official pre-trained models as a starting point to accelerate training speed and improve performance.

7. Configure training parameters: Adjust the training parameters as needed, such as batch size, learning rate, number of training rounds, etc. This can usually be done through command-line arguments or configuration files.

8. Start training: Use the command-line interface provided by YOLO11 to start the training process. For example:

   yolo train data=dataset.yaml model=yolo11n.yaml epochs=100 imgsz=640

More parameters are as follows:

|

Parameters

 ||

Default value

 ||

Description

 |

| `model` || `None` ||

Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.

 |
| `data` || `None` ||

Path to the dataset configuration file (e.g., `coco8.yaml`). This file contains dataset-specific parameters, including paths to training and validation data , class names, and number of classes.

 |
| `epochs` || `100` ||

Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.

 |
| `time` || `None` ||

Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.

 |
| `patience` || `100` ||

Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.

 |
| `batch` || `16` || Batch size

, with three modes: set as an integer (e.g., `batch=16`), auto mode for 60% GPU memory utilization (`batch=-1`), or auto mode with specified utilization fraction (`batch=0.70`).

 |
| `imgsz` || `640` ||

Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.

 |
| `save` || `True` ||

Enables saving of training checkpoints and final model weights. Useful for resuming training ormodel deployment.

 |
| `save_period` || `-1` ||

Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.

 |
| `cache` || `False` ||

Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.

 |
| `device` || `None` ||

Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=0,1`), CPU (`device=cpu`), or MPS for Apple silicon (`device=mps`).

 |
| `workers` || `8` ||

Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.

 |
| `project` || `None` ||

Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.

 |
| `name` || `None` ||

Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.

 |
| `exist_ok` || `False` ||

If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.

 |
| `pretrained` || `True` ||

Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.

 |
| `optimizer` || `'auto'` ||

Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability.

 |
| `verbose` || `False` ||

Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.

 |
| `seed` || `0` ||

Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.

 |
| `deterministic` || `True` ||

Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.

 |
| `single_cls` || `False` ||

Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.

 |
| `rect` || `False` ||

Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.

 |
| `cos_lr` || `False` ||

Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.

 |
| `close_mosaic` || `10` ||

Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.

 |
| `resume` || `False` ||

Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.

 |
| `amp` || `True` ||

Enables AutomaticMixed Precision 
 (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.

 |
| `fraction` || `1.0` ||

Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.

 |
| `profile` || `False` ||

Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.

 |
| `freeze` || `None` ||

Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning 
.

 |
| `lr0` || `0.01` ||

Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.

 |
| `lrf` || `0.01` ||

Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.

 |
| `momentum` || `0.937` ||

Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.

 |
| `weight_decay` || `0.0005` ||

L2 regularization  term, penalizing large weights to prevent overfitting.

 |
| `warmup_epochs` || `3.0` ||

Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.

 |
| `warmup_momentum` || `0.8` ||

Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.

 |
| `warmup_bias_lr` || `0.1` ||

Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.

 |
| `box` || `7.5` ||

Weight of the box loss component in the loss_function, influencing how much emphasis is placed on accurately predicting bouding box coordinates.

 |
| `cls` || `0.5` ||

Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.

 |
| `dfl` || `1.5` ||

Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.

 |
| `pose` || `12.0` ||

Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.

 |
| `kobj` || `2.0` ||

Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.

 |
| `label_smoothing` || `0.0` ||

Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.

 |
| `nbs` || `64` ||

Nominal batch size for normalization of loss.

 |
| `overlap_mask` || `True` ||

Determines whether object masks should be merged into a single mask for training, or kept separate for each object. In case of overlap, the smaller mask is overlayed on top of the larger mask during merge.

 |
| `mask_ratio` || `4` ||

Downsample ratio for segmentation masks, affecting the resolution of masks used during training.

 |
| `dropout` || `0.0` ||

Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.

 |
| `val` || `True` ||

Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.

 |
| `plots` || `False` ||

Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.

 |

Here, the `data` parameter points to your data configuration file, the `model` parameter specifies the model architecture used, the `epochs` setting determines the number of training iterations, and `imgsz` sets the size of input images.

IV. Monitoring and Assessment

9. Monitor training process: Observe the changes in loss function to ensure that the model is learning properly.

10. Model Evaluation: After training, evaluate the model's performance on the validation set, checking metrics such as mAP (Mean Average Precision).

11. Adjust hyperparameters: If the model's performance is poor, it may be necessary to adjust hyperparameters, such as increasing training rounds or changing learning rates, and retrain the model.

V. Use the model

12. Exporting the Model: After training, you can export the model as an ONNX or other format for deployment to different platforms. For example, converting PyTorch into an ONNX model can be done with the following command:
yolo export model=best.pt format=onnx

The table below provides a detailed overview of the configurations and options available for exporting YOLO models into different formats. These settings are for optimizing the performance of exported models. Size and cross-platform and environment compatibility are crucial. Correct configuration ensures that the model is ready to be deployed in the expected applications with optimal efficiency.

|

Parameters

 ||

Type

 ||

Default value

 ||

Description

 |

| `format` || `str` || `'torchscript'` ||

Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'tensorflow'`, or others, defining compatibility with various deployment environments.

 |
| `imgsz` || `int`

 or `tuple`

 || `640` ||

Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.

 |
| `keras` || `bool` || `False` ||

Enables export to Keras format for Tensorflow SavedModel, providing compatibility with TensorFlow serving and APIs.

 |
| `optimize` || `bool` || `False` ||

Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.

 |
| `half` || `bool` || `False` ||

Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.

 |
| `int8` || `bool` || `False` ||

Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.

 |
| `dynamic` || `bool` || `False` ||

Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports, enhancing flexibility in handling varying image dimensions.

 |
| `simplify` || `bool` || `True` ||

Simplifies the model graph for ONNX exports with `onnxslim`, potentially improving performance and compatibility.

 |
| `opset` || `int` || `None` ||

Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.

 |
| `workspace` || `float` || `4.0` ||

Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance.

 |
| `nms` || `bool` || `False` ||

Adds Non-Maximum Suppression (NMS) to the CoreML export, essential for accurate and efficient detection post-processing.

 |
| `batch` || `int` || `1` ||

Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.

 |
| `device` || `str` || `None` ||

Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`) or DLA for NVIDIA Jetson (`device=dla:0` or `device=dla:1`).

 |

Adjusting these parameters can customize the export process. To meet specific requirements. Assume deployment environment, Hardware constraints and performance objectives. Selecting the appropriate format and settings is crucial for achieving model size. The best balance between speed and precision is crucial.

Export format:

The available YOLO11 export format is as shown in the table below. You can export to any format using the `format` parameter, which is either `'onnx'` or `'engine'`. You can directly perform predictions or validation on the exported model, i.e., `yolo predict model=yolo11n.onnx`. After the export is completed, your model usage example will be displayed.

|

Export format

 ||

Format parameters

 ||

Model

 ||

Attribute

 ||

Parameters

 |

|

pytorch

 ||

-

 || `yolo11n.pt` ||

✅

 ||

-

 |
|

torchscript

 || `torchscript` || `yolo11n.torchscript` ||

✅

 || `imgsz`

, `optimize`, `batch`

 |
|

onnx

 || `onnx` || `yolo11n.onnx` ||

✅

 || `imgsz`

, `half`, `dynamic`, `simplify`, `opset`, `batch`

 |
| `openvino` || `openvino` || `yolo11n_openvino_model/` ||

✅

 || `imgsz`

, `half`, `int8`, `batch`

 |
|

tensorrt

 || `engine` || `yolo11n.engine` ||

✅

 || `imgsz`

, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch`

 |
|

CoreML

 || `coreml` || `yolo11n.mlpackage` ||

✅

 || `imgsz`

, `half`, `int8`, `nms`, `batch`

 |
|

TF SaveModel

 || `saved_model` || `yolo11n_saved_model/` ||

✅

 || `imgsz`

, `keras`, `int8`, `batch`

 |
|

TF GraphDef

 || `pb` || `yolo11n.pb` ||

❌

 || `imgsz`

, `batch`

 |
|

TF Lite

 || `tflite` || `yolo11n.tflite` ||

✅

 || `imgsz`

, `half`, `int8`, `batch`

 |
|

TF Edge TPU

 || `edgetpu` || `yolo11n_edgetpu.tflite` ||

✅

 || `imgsz` |
|

TF.js

 || `tfjs` || `yolo11n_web_model/` ||

✅

 || `imgsz`

, `half`, `int8`, `batch`

 |
|

PaddlePaddle

 || `paddle` || `yolo11n_paddle_model/` ||

✅

 || `imgsz`

, `batch`

 |
|

MNN

 || `mnn` || `yolo11n.mnn` ||

✅

 || `imgsz`

, `batch`, `int8`, `half`

 |
|

NCNN

 || `ncnn` || `yolo11n_ncnn_model/` ||

✅

 || `imgsz`

, `half`, `batch`

 |

13. Test Model: Test the model on new data to ensure its generalization capability is good.

The above is the basic steps to train one's own dataset using YOLO11. Please adjust the specific details in these steps according to your actual situation. Hopefully, this information will be helpful to you!

[Common Evaluation Parameters Introduction]

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

1. **Class**

: 

-

This usually refers to the categories of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

1. **Images**

: 

-

Represents the number of images in the validation set. The validation set is a dataset used to evaluate model performance, separate from the training set to ensure fairness in evaluation results.

1. **Instances**

: 

-

The total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images, each image has an average of 5 target objects, then Instances is 500.

1. **P (Precision)**

: 

-

Precision is the proportion of actual positive samples predicted by the model. The formula is: Precision = TP / (TP + FP), where TP represents true positives (True Positives) and FP represents false positives (False Positives).

1. **R (Recall)**

: 

-

Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

1. **mAP50**

: 

-

Mean Average Precision (mAP) at an IoU threshold of 0.5, where IoU is the metric used to measure the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall when evaluating model performance across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

1. **mAP50-95**

: 

-

This represents the average accuracy of the model across IoU values ranging from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and ground truth boxes. In object detection tasks, higher IoU thresholds mean that models need to be more accurate in their target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a more comprehensive assessment of the model's performance.

These indicators collectively form an important framework for evaluating the performance of target detection models. By comparing the performance of different models on these indicators, It can be determined which model may be more effective in practical applications.

[Usage Steps]

Usage Steps:
1. Install the YOLOv11 environment according to the official framework ultralytics installation tutorial, and install PyQt5.
2. After switching to the YOLOv11 environment you have installed, and switch to the source code directory, execute python main.py to start the interface, and perform the corresponding operations.

[Provided Files]

Python source code
YOLOv11n.onnx model (without PyTorch model)
Training map, P, R curves (in weights\results.png)
Test images (under test_img folder)

Note: No dataset provided.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
