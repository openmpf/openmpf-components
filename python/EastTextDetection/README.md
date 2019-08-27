# Overview

This repository contains source code and model data for the OpenMPF EAST text detection component. EAST (Efficient and Accurate Scene Text detector) was developed by Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang, and published in 2017. https://arxiv.org/abs/1704.03155

# Introduction

The component locates pieces of text in images and videos, and reports their location, angle of rotation, and text type as a single track detection. The location is expressed as the pixel coordinates of the upper left corner of the text bounding box. The angle of rotation is stored as `ROTATION` in the track's detection properties, and is expressed in positive degrees counterclockwise from the 3 o'clock position. The text type is stored as `TEXT_TYPE` in the track's detection properties, and takes either the value `STRUCTURED` or `UNSTRUCTURED`. Large amounts of uniform text is considered to be "structured", and is typically found in scans of documents and screenshots. Unstructured, or "wild" text, is usually short and isolated, and is typical of road signs.

# Properties

By default, images and videos are processed at full resolution. However, users can choose to resize inputs while retaining the same aspect ratio with the `MAX_SIDE_LENGTH` parameter. Smaller values will significantly shorten processing time, but can have a negative impact on recall.

When handling video, the component processes frames in batches. Users can control the size of these batches by setting the `BATCH_SIZE` parameter. Larger values (e.g. 32 or 64) will improve processing speed at the cost of a larger memory footprint. By default, `BATCH_SIZE` is set to 1.

Users can control how frequently text is detected with the `CONFIDENCE_THRESHOLD` property, which is the threshold value for filtering detections by bounding box confidence. By raising this parameter, fewer detections will be reported, but those detections will tend to be of higher quality. Lowering it will have the opposite effect; many bounding boxes will be returned, but many of them may represent false positives, and contain no text in truth. By default, this property is set to 0.8, which works well for Latin script (used in most western European languages).

The EAST model naturally prefers horizontal text, as text at high degrees of rotation is rare in unstructured reading datasets. If the user anticipates a large amount of rotated text, they can indicate with the `ROTATE_AND_DETECT` parameter that the input should be processed a second time after rotating 90 degrees.

The `SUPPRESS_VERTICAL` property controls whether to discard detections which are taller than they are wide. When dealing with English text, such detections are usually incorrect, and when `ROTATE_AND_DETECT` is `TRUE`, these spurious bounding boxes will often interfere with correct detections.

Note that `SUPPRESS_VERTICAL=TRUE` will result in missed detections for language read from top to bottom, as is common in some styles of East Asian scripts. If vertical text is expected, it is recommended to set this property to `FALSE`, in addition to lowering the `CONFIDENCE_THRESHOLD` property to small values like `0.001`, as EAST does not handle this type of text well natively.

The `MERGE_REGIONS` property dictates how detections are filtered and/or combined. When set to `FALSE`, standard non-maximum suppression is used with `NMS_MIN_OVERLAP` as a threshold value. In this case, when the ratio between the areas of intersection and union of two regions is greater than or equal to `NMS_MIN_OVERLAP`, the detection with the lower confidence score is dropped.

When `MERGE_REGIONS` is `TRUE`, detections are merged (replaced with a minimal enclosing bounding box at the same orientation) rather than filtered. Users can control how aggressively the component merges regions of text with the `MERGE_MIN_OVERLAP`, `MERGE_MAX_TEXT_HEIGHT_DIFFERENCE`, and `MERGE_MAX_ROTATION_DIFFERENCE` parameters. For two detections to be merged, all of the following must be true:
1. The ratio between the area of intersection of two regions and the area of the smaller region is greater or equal to than `MERGE_MIN_OVERLAP`. For example, if `MERGE_MIN_OVERLAP = 0.5`, then the larger region must cover at least half of the smaller region.
2. The relative difference in detected text height for two regions is less than or equal to `MERGE_MAX_TEXT_HEIGHT_DIFFERENCE`.
3. The absolute difference (in degrees) between the orientations of two regions is less than or equal to `MERGE_MAX_ROTATION_DIFFERENCE`.

Note that `MERGE_MIN_OVERLAP` and `NMS_MIN_OVERLAP` work with different measures of "overlap"; in standard non-max-suppression, the intersect over union (IoU) between boxes is used, while our merging algorithm compares the area of intersect over the smaller area (IoSA).

There are two properties related to padding. `TEMPORARY_PADDING` controls the temporary padding added to bounding boxes during non-maximum suppression and merging; this can ensure that adjacent detections overlap sufficiently to be merged or filtered. After the NMS or merge step is completed, this padding is effectively removed. To control the padding on output regions, use the `FINAL_PADDING` property. This value has no effect on either non-maximum suppression or the merge step. In the case of both properties, the padding is applied symmetrically to all four sides of bounding boxes, and is equal to the supplied value multiplied by an estimation of the text height. Thus, a padding property value of `0.5` will increase the width and height of boxes by roughly the line height of the text they contain. 

The `MIN_STRUCTURED_TEXT_THRESHOLD` property determines when text is considered to be structured or unstructured, and is compared with the average confidence score across the entire image. In images with structured text, this mean confidence will tend to be higher, as a larger percentage of the image is likely to be occupied by text. Larger values of `MIN_STRUCTURED_TEXT_THRESHOLD` will result in less text being marked as structured.
