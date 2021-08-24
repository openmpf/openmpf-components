# Overview

This repository contains source code for the MPF OpenCV Yolo detection component.
This component detects objects in images and videos using the OpenCV Yolo Detector.  The component can perform inferencing using OpenCV's DNN libraries, or rely on an external [Triton Inference Server](https://github.com/triton-inference-server).

## Trained Models

The OpenCV Yolo detection component comes with MS COCO trained YoloV4 model:

* Model files,
[yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg) and
[yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT).
This model was trained on MS COCO data set. For details of the YoloV4 model please see
[Bochkovskiy, A et al.](https://arxiv.org/abs/2004.10934), *"YOLOv4: Optimal Speed
and Accuracy of Object Detection"*, 2020.

* Triton YOLOv4 model implementation is optimized for a Triton Inference Server (e.g. 2.11.0) compiled with a specific version of TensorRT (e.g. r21.06) for specific GPU hardware. An associated plugin file (liblayerplugin.so) is needed for YOLO layer implementation on the inference server and an engine file (yolov4.engine) for model deployment.  These model engine and plugin files are not included in this repo, but can be created as described [here](https://github.com/isarsoft/yolov4-triton-tensorrt) by isarsoft. Note that they are image size and inference batch size specific.

## Algorithms Used

Both [OpenCV](https://opencv.org) and [DLIB](http://dlib.net) algorithms are used, as are the [Triton Inference Server](https://github.com/triton-inference-server) client libraries. Of note:

* A linear assignment cost solver, a DLIB implementation of the Hungarian algorithm (also
  known as the Kuhn-Munkres algorithm) to solve detection to track assignment. See
  [Kuhn](https://doi.org/10.1002/nav.3800020109), *"The Hungarian Method for the
  assignment problem"*, 1955 for original method.

## Tracking Overview

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
* Any tracks that did not receive a detection are then optionally continued using an OpenCV correlation
  tracker predicted bounding box as a detection stand-in if possible. If this occurs, a detection
  property named `"FILLED_GAP"` with value `"TRUE"` will be added to the new detection. The
  detection will have a slightly lower confidence than the detection that precedes it, and it
  will inherit the classification(s) of that detection.  Use of the OpenCV tracker will incur a significant speed penalty, which is especially apparent is high throughput setups using an inference server.
* Unassigned detections become new tracks.
* All active tracks then have their object bounding box positions corrected and predicted forward
  in time with a Kalman filter for the next tracking iteration. During the assignment stages,
  potential detection-to-track assignments that produce an excessive Kalman filter residual error
  are rejected.

## Future Research

* Test Kalman Filter in isolation to figure out the best values for the noise inputs. Among other
  things, the values are dependent on the size of the objects being tracked (which is related to
  the network input size), and the frame interval.
    * Different network input sizes and frame intervals might require very different noise inputs
      to work well. If the Kalman Filter diverges frequently then tracks will get split up more.
    * Potentially expose a setting to switch between continuous and piecewise noise.
    * Potentially expose a setting to disable velocity and acceleration entries in the state
      transition matrix for width and height if the bounding boxes are too bouncy.
* Test DFT pixel similarity metric in isolation to figure out the best threshold and ensure
  desired behavior.
    * In general, the IoU distance is most likely responsible for the vast majority of track
      assignments when processing every frame of a video. Region overlap may occur less frequently
      when using higher frame intervals, in which case the tracker will fall back to DFT pixel
      similarity more. Very little testing has been done with frame intervals > 1.
* Currently, only the full and Tiny YoloV3 models are configured with a confusion matrix.
  The matrix was originally created based on testing Tiny YoloV3, so it may not be appropriate
  for use with full YoloV3. Consider generating new confusion matrices for the full and Tiny
  YoloV4 models.
