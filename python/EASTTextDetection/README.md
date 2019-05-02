# Overview

This repository contains source code and model data for the OpenMPF EAST text detection component.

# Introduction

The component locates pieces of text in images and videos, and reports their
location and angle of rotation as a single track detection. The location is
expressed as the pixel coordinates of the upper left corner of the text
bounding box. The angle of rotation is stored as ROTATION in the track's
detection properties, and is expressed in positive degrees counterclockwise
from the 3 o'clock position.

# Properties

By default, images and videos are processed at full resolution. However, users
can choose to resize inputs while retaining the same aspect ratio with the
MAX_SIDE_LENGTH parameter. Smaller values will significantly shorten processing
time, but can have a negative impact on recall.

When handling video, the component processes frames in batches. Users can
control the size of these batches by setting the BATCH_SIZE parameter. Larger
values (e.g. 32 or 64) will improve processing speed at the cost of a larger
memory footprint. By default, BATCH_SIZE is set to 1.

The EAST model naturally prefers horizontal text, as text at high degrees of
rotation is rare in unstructured reading datasets. If the user anticipates a
large amount of rotated text, they can indicate with the ROTATE_ON parameter
that the input should be processed a second time after rotating 90 degrees.

Users can control how frequently text is detected. CONFIDENCE_THRESHOLD is the
threshold value for filtering detections by bounding box confidence. By raising
this parameter, fewer detections will be reported, but those detections will
tend to be of higher quality. Lowering it will have the opposite effect; many
bounding boxes will be returned, but many of them may represent false
positives, and contain no text in truth. By default, this property is set to
0.8.

Users can control how aggressively the component merges regions of text with the OVERLAP_THRESHOLD, TEXT_HEIGHT_THRESHOLD, and ROTATION_THRESHOLD parameters. For two detections to be merged, all of the following must be true:
1. The ratio between the area of intersection of two regions and the area of the smaller region is greater than OVERLAP_THRESHOLD. For example, if OVERLAP_THRESHOLD=0.5, then the larger region must cover at least half of the smaller region.
2. The relative difference in detected text height for two regions is below TEXT_HEIGHT_THRESHOLD.
3. The absolute difference (in degrees) between the orientations of two regions is below ROTATION_THRESHOLD.
