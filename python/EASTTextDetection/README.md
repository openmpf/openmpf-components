# Overview

This repository contains source code and model data for the OpenMPF EAST text detection component.

The component locates pieces of text in images and videos, and reports their location and angle of rotation as a single track detection. The location is expressed as the pixel coordinates of the upper left corner of the text bounding box. The angle of rotation is stored as ROTATION in the track's detection properties, and is expressed in positive degrees counterclockwise from the 3 o'clock position.

By default, images and videos are processed at full resolution. However, users can choose to resize inputs while retaining the same aspect ratio with the MAX_SIDE_LENGTH parameter. Smaller values will significantly shorten processing time, but can have a negative impact on recall.

When handling video, the component processes frames in batches. Users can control the size of these batches by setting the BATCH_SIZE parameter. Larger values (e.g. 32 or 64) will improve processing speed at the cost of a larger memory footprint. By default, BATCH_SIZE is set to 1.

The EAST model naturally prefers horizontal text, as text at high degrees of rotation is rare in unstructured reading datasets. If the user anticipates a large amount of rotated text, they can indicate with the ROTATE_ON parameter that the input should be processed a second time after rotating 90 degrees.

Users can control how frequently text is detected by altering the CONFIDENCE_THRESHOLD and NMS_THRESHOLD parameters. CONFIDENCE_THRESHOLD is the threshold value for filtering detections by bounding box confidence. By raising this parameter, fewer detections will be reported, but those detections will tend to be of higher quality. Lowering it will have the opposite effect; many bounding boxes will be returned, but many of them may represent false positives, and contain no text in truth. By default, this property is set to 0.8.

NMS_THRESHOLD is a parameter for non-maximum suppression. Increasing this value will result in the component returning fewer overlapping bounding boxes. By default, this parameter is set to 0.2, meaning bounding boxes will be ignored if the ratio between the intersect and union with another box of higher confidence is greater than 0.2.
