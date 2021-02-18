# Overview

This repository contains source code for the MPF OpenCV Yolo detection component. This component detects objects in images and videos using
the OpenCV Yolo Detector.

## Trained Models

The OpenCV Yolo detection component comes with MS COCO trained YoloV4 model:

* Model files, [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg) and [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT). This model was trained on MS COCO data set. For details of the YoloV4 model please see [Bochkovskiy, A et al.](https://arxiv.org/abs/2004.10934), *"YOLOv4: Optimal Speed and Accuracy of Object Detection"*, 2020

## Algorithms used
Both [OpenCV](https://opencv.org) and [DLIB](http://dlib.net) algorithms are used. Of note:

* A linear assignment cost solver, a DLIB implementation of the Hungarian algorithm (also known as the Kuhn-Munkres algorithm) to solve detection to track assignment. See [Kuhn](https://doi.org/10.1002/nav.3800020109), *"The Hungarian Method for the assignment problem"*, 1955 for original method.

## Tracking overview
Detections in videos are aggregated into tracks as frames are processed. This is done using 
multiple object-to-track assignment stages, each using a linear assignment cost solver:
* While processing each frame, tracks that have not had any detections for some number of frames 
  are terminated.
* Detections can only be assigned to the remaining tracks that have a similar Yolo 
  object-class-group as defined by a confusion matrix 
  [see here](https://github.com/whynotw/YOLO_metric))
* A cheap intersection over union (IoU) cost based assignment is then performed which requires a 
  minimum IoU score between a track tail and a detection.
* Any remaining unassigned detections are then attempted to be assigned to tracks using an 
  absolute-pixel-difference based cost assignment. The cost is calculated _after_ aligning 
  detection and track bounding box pixel content. This registration is performed using 
  Fast Fourier transform (FFT) phase correlation conveniently available in OpenCV.
* Any tracks that did not receive a detection are then continued using an OpenCV correlation 
  tracker predicted bounding box as a detection stand-in if possible. If this occurs, a detection 
  property named `"FILLED_GAP"` with value `"TRUE"` will be added to the new detection.
* Unassigned detections become new tracks.
* All active tracks then have their object bounding box positions corrected and predicted forward 
  in time with a Kalman filter for the next tracking iteration. During the assignment stages, 
  potential detection-to-track assignments that produce an excessive Kalman filter residual error 
  are rejected.
