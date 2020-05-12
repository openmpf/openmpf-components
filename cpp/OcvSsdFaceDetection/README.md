# Overview

This repository contains source code for the MPF OpenCV SSD face detection component.  This component detects faces in images and videos using
the OpenCV Single Shot Detector.

   

## Trained Models and Algorithms used

The OpenCV SSD face detection component comes with three trained models:
<br>
* A discretized tensorflow version of OpenCV's Single Shot Detector (SSD) for faces, [opencv\_face\_detector\_uint8.pbtxt](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/opencv_face_detector.pbtxt) and [opencv\_face\_detector\_uint8.pb](https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb)\. This model was discretized from an existing caffee model\, \[opencv\_face\_detector\.caffemodel\]\(https://github\. com/opencv/opencv\_3rdparty/raw/dnn\_samples\_face\_detector\_20170830/res10\_300x300\_ssd\_iter\_140000\.caffemodel\)\. More details about how this model was trained can be found [here](https://github.com/opencv/opencv/blob/3.4.3/samples/dnn/face_detector/how_to_train_face_detector.txt). Details of the SSD model architecture are presented in "SSD: Single Shot MultiBox Detector", 2016 [Liu et al.](https://arxiv.org/abs/1512.02325)
* A 5 landmark detector model for faces, [shape\_predictor\_5\_face\_landmarks.dat](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) from DLIB and trained by its author on 7198 faces, see [here](https://github.com/davisking/dlib-models/blob/master/README.md#shape_predictor_5_face_landmarksdatbz2). This model is used to align faces prior to feature generation. It is an implementation of the Ensemble of Regression Trees (ERT) presented in "One Millisecond Face Alignment with an Ensemble of Regression Trees" 2014 by [Kazemi and Sullivan](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html).
* A face feature generator, [nn4.small2.v1.t7](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7) from [OpenFace](https://cmusatyalab.github.io/openface/). This model was tranied with publicly-available face recognition datasets based on names: FaceScrub and CASIA-WebFace and generates a 128 dimensional embedding for faces. It is used in the tracking portion of the MPF component. More details about OpenFace can be found in "Openface: A general-purpose face recognition library with mobile applications" 2016 [Amos, Ludwiczuk and Satyanarayanan](http://elijah.cs.cmu.edu/DOCS/CMU-CS-16-118.pdf)
<br>
Both [OpenCV](https://opencv.org/) and [DLIB](http://dlib.net/) algorithms are used in MPF. This component specifically uses:
<br>
* A Minimum Output Sum of Squared Error (MOSSE) tracker implemented in OpenCV's tracking API based on "Visual object tracking using adaptive correlation filters" 2010 [Bolme et al.](https://ieeexplore.ieee.org/document/5539960)
* A linear assignment cost solver, a DLIB implementation of the Hungarian algorithm (also know as the Kuhn-Munkres algorithm) to solve detection to track assignment. See "The Hungarian Method for the assignment problem", 1955 [Kuhn](https://en.wikipedia.org/wiki/Naval_Research_Logistics_Quarterly) for original method.

## Tracking implementation
In videos detected faces are aggregated into tracks as frames are processed using a linear assignment cost solver in multiple stages. While processing each frame, tracks that have not had any detections of some number of frames are terminated, then detections are assigned to the remaing tracks using Intersection Over Union (IoU) as a cost, any unassigned detections are attempted to be assigned to tracks using the cos distance with the OpenFace features. Any tracks that did not receive a detection are then continued using a MOSSE correlation tracker prediction.<br>
Since detection and feature computation are reasonably expensive operations, it can be specified that those operations are only perforemd every so often and the faster MOOSE tracker will have to fill in the gaps.  The gained speed comes at the potential cost of missed detections and more track fragmantation.

- - -

### Note

The component was coded using [Visual Studio Code](https://code.visualstudio.com/) in a docker container.  The `.vscode` and `.devcontainer` subdirectories contain setting for the development tool but are in no way required to use or build the project.