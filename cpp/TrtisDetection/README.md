# Overview

This repository contains source code for the OpenMPF TensorRT Inference Server
(TRTIS) detection component.

As a docker-only component, users must build and launch both the `trtis-detection` and `trtis-detection-server` services
from `docker-compose.components.yml` as part of the `openmpf-docker` [deployment](https://github.com/openmpf/openmpf-docker/tree/develop). These two services
will build and host the MPF TRTIS client and the NVIDIA TRTIS server from their respective Dockerfile images:
`TrtisDetection/Dockerfile` and `TrtisDetection/trtserver_dockerfile/Dockerfile`.

Users can refer to the
[docker-compose.components.yml](https://github.com/openmpf/openmpf-docker/blob/develop/docker-compose.components.yml)
file to launch custom deployments with this component.
Custom TRTIS server model directory locations can also be specified through the `docker-compose.components.yml` file under the `trtis-detection-server` service.

The TRTIS detection component currently supports a modified version of the [Faster R-CNN model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
(`faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28`)
which is trained on the Common Objects in Context (COCO) dataset. Labels corresponding to each COCO class can be found
in `TrtisDetection/plugin-files/models/ip_irv2_coco/ip_irv2_coco.labels`.

Currently, the component supports both image and video jobs.

Users can control inference and model behavior through the following job parameters:

* `TRTIS_SERVER` : Specifies the DNS name or IP address and GRPC port of an NVIDIA TensorRT Inference Server (TRTIS). If the value is set to the empty string "", the component will instead use the environment variable for `TRTIS_SERVER`, if it is available.

* `MODEL_NAME` : Specifies the model on `TRTIS_SERVER` to be used for inferencing. Currently only the default model, `ip_irv2_coco`, is supported.

* `MODEL_VERSION`: Specifies the version number of the inference server model, with the default value of `-1` indicating usage of the latest available version on the server.

* `MAX_INFER_CONCURRENCY` : Specifies the maximum number of inference requests that will be sent to the server concurrently for video frame inferencing. If the value is the empty string "", the component will instead use the environment variable for `MAX_INFER_CONCURRENCY`, if it is available.

* `USER_FEATURE_X_LEFT_UPPER` and `USER_FEATURE_Y_LEFT_UPPER` allow users to define the upper left coordinates for a custom bounding box of the image or video frame during inference. `USER_FEATURE_WIDTH` and `USER_FEATURE_HEIGHT` allow users to control the width and height of the custom bounding box.

* `CLIENT_PRESCALING_ENABLE`: Toggles whether to scale images and video frames to the approximate `[1024 x 600]` image size required by the `ip_irv2_coco` model.

For enabling generation of similarity features:

* `USER_FEATURE_ENABLE` : Toggles generation of similarity feature detections for a user-specified bounding box.

* `CLASS_FEATURE_ENABLE`: Toggles generation of similarity features for any COCO-class objects detected.

* `FRAME_FEATURE_ENABLE`: Toggles generation of a size-weighted average of all other features found in an image or frame.

* `EXTRA_FEATURE_ENABLE`: Toggles generation of similarity features for candidate object regions that could not be classified as COCO objects.

* `EXTRA_CONFIDENCE_THRESHOLD`: Specifies threshold for object region detections that could not be classified as COCO objects. Please note that confidence scores for these extra detections is generally extremely low.

For object tracking in video frames, the following settings control how objects are assessed and organized within the MPF video tracks.
Each condition must be met for an object to be considered to be part of the same MPF video track:

* `TRACK_MAX_FEATURE_GAP`: Specifies the maximum gap in the similarity feature space, based on the cosine or inner product distance.

* `TRACK_MAX_FRAME_GAP`: Specifies the maximum gap in video frames.

* `TRACK_MAX_SPACE_GAP`: Specifies the maximum gap in normalized pixel space.

Currently video tracking uses basic search space calculations to match preexisting tracks to the closest candidate object detections being considered within the current video frame. If a match occurs, an MPFImageLocation is appended to the matching track.
When a newly detected object fails to match to a preexisting track, a new track is created for that object.

