# Overview

This repository contains source code for the MPF OpenCV SSD face detection component.  This component detects faces in images and videos using
the OpenCV Single Shot Detector.

## Trained Models

The OpenCV SSD face detection component comes with three trained models:

* A discretized tensorflow version of OpenCV's Single Shot Detector (SSD) for faces, [opencv_face_detector_uint8.pbtxt](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/opencv_face_detector.pbtxt) and [opencv_face_detector_uint8.pb](https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb). This model was discretized from an existing Caffe model, [opencv_face_detector.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel). More details about how this model was trained can be found [here](https://github.com/opencv/opencv/blob/3.4.3/samples/dnn/face_detector/how_to_train_face_detector.txt). Details of the SSD model architecture are presented in [Liu et al.](https://arxiv.org/abs/1512.02325), *"SSD: Single Shot MultiBox Detector"*, 2016.
* A 5-landmark detector model for faces, [shape_predictor_5_face_landmarks.dat](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) from DLIB and trained by its author on 7198 faces, see [here](https://github.com/davisking/dlib-models/blob/master/README.md#shape_predictor_5_face_landmarksdatbz2). This model is used to align faces prior to feature generation. It is an implementation of the Ensemble of Regression Trees (ERT) presented by [Kazemi and Sullivan](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html), *"One Millisecond Face Alignment with an Ensemble of Regression Trees"*, 2014.
* A face feature generator [Torch](http://torch.ch/) model, [nn4.small2.v1.t7](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7) from [OpenFace](https://cmusatyalab.github.io/openface/). This model was trained with publicly-available face recognition datasets based on names: FaceScrub and CASIA-WebFace. It generates a 128-dimensional embedding for faces. It is used in the tracking portion of the OpenMPF component. More details about OpenFace can be found in [Amos, Ludwiczuk and Satyanarayanan](http://elijah.cs.cmu.edu/DOCS/CMU-CS-16-118.pdf), *"Openface: A general-purpose face recognition library with mobile applications"*, 2016.

## Algorithms Used
Both [OpenCV](https://opencv.org) and [DLIB](http://dlib.net) algorithms are used in OpenMPF. This component specifically uses:

* A Minimum Output Sum of Squared Error (MOSSE) tracker implemented in OpenCV's tracking API based on [Bolme et al.](https://ieeexplore.ieee.org/document/5539960), *"Visual object tracking using adaptive correlation filters."*, 2010.
* A linear assignment cost solver, a DLIB implementation of the Hungarian algorithm (also know as the Kuhn-Munkres algorithm) to solve detection to track assignment. See [Kuhn](https://doi.org/10.1002/nav.3800020109), *"The Hungarian Method for the assignment problem"*, 1955 for original method.

## Tracking Implementation
Detected faces in videos are aggregated into tracks as frames are processed. This is done using multiple stages and linear assignment cost solver: While processing each frame, tracks that have not had any detections for `TRACKING_MAX_FRAME_GAP + 1` frames are terminated, then detections are assigned to the remaining tracks using Intersection Over Union (IoU) as a cost. Any unassigned detections are attempted to be assigned to tracks using the cos distance with the OpenFace features. Any tracks that did not receive a detection are then continued using a correlation tracker prediction. Any unassigned new detections become new tracks.

Since detection and feature computation are computationally expensive operations, by default they are not performed on every frame. The interval can be configured by setting the `DETECTION_FRAME_INTERVAL` property. We rely on the faster correlation tracker to fill in the gaps for the frames that are skipped. The processing speed gained by doing this comes at the cost of potentially missed detections and more track fragmentation.

- - -

### Note

The component was coded using [Visual Studio Code](https://code.visualstudio.com) in a docker container.  The `.vscode` and `.devcontainer` subdirectories contain setting for the development tool but are in no way required to use or build the project.
