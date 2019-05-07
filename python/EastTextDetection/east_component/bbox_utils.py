#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2019 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################
import cv2
import numpy as np

# A vectorized function for getting the area of QUAD-formatted bounding boxes
contour_area = np.vectorize(
    lambda b: cv2.contourArea(b.astype(np.float32)),
    signature='(4,2)->()'
)

# A vectorized function for getting the area of intersection between
#  QUAD-formatted bounding boxes
contour_intersect_area = np.vectorize(
    lambda b0,b1: cv2.intersectConvexConvex(b0.astype(np.float32),b1.astype(np.float32))[0],
    otypes=[float,np.ndarray],
    signature='(4,2),(4,2)->()'
)

def rbox_to_quad(geoms):
    """ Convert RBOX geometry to bounding box corner coordinates
    RBOX format:
        [x, y, dt, dr, db, dl, rotation]
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]]
    """
    # Get cosine and sine of the box angles
    cs = np.hstack((np.cos(geoms[:,[6]]), np.sin(geoms[:,[6]])))

    quads = np.empty((geoms.shape[0],4,2),dtype=np.float32)

    # Translate to the left and right edges
    quads[:,[0,3],:] = (geoms[:,[0,1]] + geoms[:,[5]] * cs * [-1, +1])[:,None,:]
    quads[:,[1,2],:] = (geoms[:,[0,1]] + geoms[:,[3]] * cs * [+1, -1])[:,None,:]

    # Translate to the top and bottom edges
    quads[:,(0,1),:] -= (cs * geoms[:,[2]])[:,None,::-1]
    quads[:,(2,3),:] += (cs * geoms[:,[4]])[:,None,::-1]

    return quads

def quad_to_iloc(quads, scores):
    """ Convert QUAD geometry to OpenMPF ImageLocation format
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]]
     OpenMPF ImageLocation format:
         [tl_x, tl_y, w, h, rotation, score]
    """
    w_vec = quads[:,1] - quads[:,0]
    h_vec = quads[:,3] - quads[:,0]

    ilocs = np.empty((quads.shape[0],6), dtype=np.float32)
    ilocs[:,[0,1]] = quads[:,0]
    ilocs[:,2] = np.sqrt(np.sum(w_vec*w_vec, -1))
    ilocs[:,3] = np.sqrt(np.sum(h_vec*h_vec, -1))
    ilocs[:,4] = (720 - np.degrees(np.arctan2(-w_vec[:,1], w_vec[:,0]))) % 360
    ilocs[:,5] = scores
    return ilocs


def nms(quads, scores, overlap_threshold):
    areas = contour_area(quads)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        js = order[1:]
        inter = contour_intersect_area(quads[i], quads[js])
        union = areas[i] + areas[js] - inter
        iou = inter / union
        inds = np.flatnonzero(iou <= overlap_threshold)
        order = order[inds+1]
    return keep

def merge(quads, rotation):
    """ Merge a group of QUAD-formatted bounding boxes, enclosing them in a
        minimal bounding rectangle with the specified rotation, which should
        be given in radians above the 3 o'clock position.
    """
    # Create rotation matrix
    c, s = np.cos(rotation), np.sin(rotation)
    rot = np.array([[c,-s],[s,c]])

    # Rotate the points
    proj = quads.reshape((-1,2)).dot(rot.transpose())

    # Get the extreme coordinates
    merged = np.empty((4,2),dtype=np.float32)
    merged[[0,1],:] = proj.min(0)
    merged[[2,3],:] = proj.max(0)
    merged[1,0] = merged[2,0]
    merged[3,0] = merged[0,0]

    # Rotate the points back
    return merged.dot(rot)


class MergedRegions(object):
    def __init__(self, rboxes, scores):
        self.quads = rbox_to_quad(rboxes)
        self.rotations = rboxes[:,6]
        self.heights = rboxes[:,2] + rboxes[:,4]
        self.areas = contour_area(self.quads)
        self.weights = scores
        self.scores = scores

def merge_pass(regions, overlap_threshold, text_height_threshold,
               rotation_threshold):
    """ Complete one merge pass. This takes one box at a time (largest area
        first) and merges it with all eligible boxes. This merged box is added
        to a list, and all constituent boxes removed from the original list.
        The largest remaining box is then taken, and the process is repeated
        until the original list is empty.
    """
    # Sort by bounding box area
    order = np.argsort(regions.areas)[::-1]
    quads = regions.quads[order]
    rotations = regions.rotations[order]
    heights = regions.heights[order]
    weights = regions.weights[order]
    scores = regions.scores[order]
    areas = regions.areas[order]

    merged_quads = []
    merged_rotations = []
    merged_heights = []
    merged_weights = []
    merged_scores = []
    merged_any = False
    while len(quads) > 0:
        inter = contour_intersect_area(
            quads[0].astype(np.float32),
            quads[1:].astype(np.float32)
        )
        # Divide by smaller area, not the union
        to_merge = (inter / areas[1:]) > overlap_threshold

        # Filter out boxes whose text height does not match
        h_dist = 2 * np.abs(heights[0]-heights[1:]) / (heights[0]+heights[1:])
        height_matches = h_dist < text_height_threshold
        to_merge = np.logical_and(to_merge, height_matches)

        # Filter out boxes whose rotation does not match
        r_dist = np.abs(rotations[0] - rotations[1:])
        rotation_matches = r_dist < (rotation_threshold * np.pi / 180)
        to_merge = np.logical_and(to_merge, rotation_matches)

        # If no boxes can be merged with this one, pop it off and continue
        if not np.any(to_merge):
            merged_quads.append(quads[0])
            merged_rotations.append(rotations[0])
            merged_heights.append(heights[0])
            merged_weights.append(weights[0])
            merged_scores.append(scores[0])
            quads = quads[1:]
            rotations = rotations[1:]
            heights = heights[1:]
            weights = weights[1:]
            scores = scores[1:]
            areas = areas[1:]
            continue

        # Merge the eligible boxes in the tail with the head box
        to_merge = np.hstack((True, to_merge))

        # The rotation and text height for a merged region is a weighted
        #  average of the constituent boxes' rotations and text heights
        merging_weights = weights[to_merge]
        scale = 1.0 / merging_weights.sum()
        merged_rotation = np.sum(rotations[to_merge] * merging_weights) * scale
        merged_height = np.sum(heights[to_merge] * merging_weights) * scale

        # The weight for a merged region is the sum of constituent weights
        merged_weight = merging_weights.sum()

        # The score for a merged region is the maximum score of constituents
        merged_score = scores[to_merge].max()

        # Merge the boxes at the weighted average rotation
        merged_quad = merge(quads[to_merge], merged_rotation)

        # Save the merged box
        merged_quads.append(merged_quad)
        merged_rotations.append(merged_rotation)
        merged_heights.append(merged_height)
        merged_weights.append(merged_weight)
        merged_scores.append(merged_score)

        # Continue to merge together all the boxes that were not just merged
        unmerged = np.logical_not(to_merge)
        quads = quads[unmerged]
        rotations = rotations[unmerged]
        heights = heights[unmerged]
        weights = weights[unmerged]
        areas = areas[unmerged]
        scores = scores[unmerged]

        merged_any = True

    # Update the regions structure in place
    regions.quads = np.stack(merged_quads)
    regions.rotations = np.array(merged_rotations)
    regions.heights = np.array(merged_heights)
    regions.weights = np.array(merged_weights)
    regions.scores = np.array(merged_scores)
    regions.areas = contour_area(regions.quads.astype(np.float32))

    return merged_any

def merge_regions(rboxes, scores, overlap_threshold, text_height_threshold,
                  rotation_threshold):
    """ An approximate locality-aware variant of non-maximum suppression, which
        merges together overlapping boxes rather than suppressing them.

        Non-maximum suppression (NMS) selects regions in order of decreasing
        confidence and discards all lower-confidence regions whose intersection
        over union with the current region is greater than some threshold value.
        This has the effect of pruning overlapping boxes from the set of
        detections, retaining only those with the highest confidence scores.

        Locality-Aware NMS (LANMS), proposed by Zhou et al. in EAST: An
        Efficient and Accurate Scene Text Detector, performs an initial linear
        time row-merging pass which significantly reduces the number of
        detections before performing the standard quadratic time NMS procedure.
        In the row-merging pass, adjacent overlapping regions are combined via a
        confidence-weighted average of their geometries. This is done
        iteratively, so the combined region is then compared with the next
        adjacent detection.

        We make several key changes to the LANMS algorithm.
        1. The initial row-merging pass is not iterative: we
           instead take "runs" of overlapping bounding boxes (essentially
           connected components in the flattened grid) and merge them together.
           This allows for a significant speedup through NumPy vectorization.
        2. In the standard LANMS scheme, "merging" means to take the
           score-weighted average of bounding box corner coordinates. In our
           algorithm, we replace overlapping regions with a minimal oriented
           bounding box which encloses the set of regions to be merged, whose
           rotation is defined as the score-weighted average of the constituent
           bounding box orientations.
        3. Even after the locality-aware first pass, we merge boxes rather than
           suppress the ones with lower confidence. As above, the orientation
           of the merged box is a confidence-weighted average.
        4. After the first pass, we threshold not the intersect over union, but
           the intersect over smallest area. This is because, as merged regions
           grow larger, the IOU between them and smaller boxes will shrink.
           Because the threshold value is constant, large boxes will eventually
           never merge with smaller boxes.
        5. In addition to thresholding the overlap, we theshold the similarity
           between regions in rotation and text height. This way, text with very
           dissimilar orientations or scales will not be combined. The text
           height at the first pass is defined as the height of the bounding
           box. When merged, the merged region's text height is the
           confidence-weighted average of the text heights of the constituent
           boxes (so a large region composed of many small boxes will still be
           associated with small text).

        Note: During merging, the confidence "weight" of a merged box is the sum
           of the constituent confidences. This is to prevent a small box
           significantly altering the properties of a large merged region
           containing many constituent boxes. However, the reported confidence
           of a merged region is the maximum of its constituent scores.
    """
    regions = MergedRegions(rboxes, scores)

    # Do the initial locality-aware pass
    quads = regions.quads
    rotations = regions.rotations
    heights = regions.heights
    weights = regions.weights
    scores = regions.scores
    areas = regions.areas

    inter = contour_intersect_area(
        quads[:-1].astype(np.float32),
        quads[1:].astype(np.float32)
    )
    # Divide by smaller area, not the union
    to_merge = inter / np.minimum(areas[:-1], areas[1:]) > overlap_threshold

    # Filter out boxes whose text height does not match
    h_dist = 2 * np.abs(heights[:-1]-heights[1:]) / (heights[:-1]+heights[1:])
    height_matches = h_dist < text_height_threshold
    to_merge = np.logical_and(to_merge, height_matches)

    # Filter out boxes whose rotation does not match
    r_dist = np.abs(rotations[:-1] - rotations[1:])
    rotation_matches = r_dist < (rotation_threshold * np.pi / 180)
    to_merge = np.logical_and(to_merge, rotation_matches)

    if np.any(to_merge):
        # Here, to_merge is the boolean map indicating whether the box at each
        #  index can be merged with the one directly after it.
        to_merge = np.hstack((to_merge, False))

        # Break down into runs of True cells. These runs can all be merged
        #  together at once. This is the approximation; with true locality
        #  aware NMS, boxes are merged successively, and the merging is done
        #  by averaging box corner coordinates. Here, we merge by taking a
        #  large box which encloses runs of overlapping constituents
        diffs = np.diff(np.hstack((False, to_merge, False)).astype(np.int8))
        run_starts = np.flatnonzero(diffs > 0)
        run_ends = np.flatnonzero(diffs < 0)

        merged_quads = []
        merged_rotations = []
        merged_heights = []
        merged_weights = []
        merged_scores = []
        for i0,i1 in zip(run_starts, run_ends):
            # The rotation and text height for a merged region is a weighted
            #  average of the constituent boxes' rotations and text heights
            merging_weights = weights[i0:i1]
            scale = 1.0 / merging_weights.sum()
            merged_rotation = np.sum(rotations[i0:i1] * merging_weights) * scale
            merged_height = np.sum(heights[i0:i1] * merging_weights) * scale

            # The weight for a merged region is the sum of constituent weights
            merged_weight = merging_weights.sum()

            # The score for a merged region is the maximum score of constituents
            merged_score = scores[i0:i1].max()

            # Merge the boxes at the weighted average rotation
            merged_quad = merge(quads[i0:i1], merged_rotation)

            # Save the merged box
            merged_quads.append(merged_quad)
            merged_rotations.append(merged_rotation)
            merged_heights.append(merged_height)
            merged_weights.append(merged_weight)
            merged_scores.append(merged_score)

        # The first in a run of False cells was merged with the previous run
        to_merge[1:] += to_merge[:-1]

        # Update the structure to include the merged runs as well as the
        #  original boxes left unmerged
        unmerged = np.logical_not(to_merge)
        regions.quads = np.vstack((quads[unmerged], np.stack(merged_quads)))
        regions.rotations = np.hstack((rotations[unmerged], merged_rotations))
        regions.heights = np.hstack((heights[unmerged], merged_heights))
        regions.weights = np.hstack((weights[unmerged], merged_weights))
        regions.areas = contour_area(regions.quads.astype(np.float32))
        regions.scores = np.hstack((scores[unmerged], merged_scores))

    # Continue until all eligible boxes have been merged
    while True:
        merged_any = merge_pass(
            regions,
            overlap_threshold=overlap_threshold,
            text_height_threshold=text_height_threshold,
            rotation_threshold=rotation_threshold
        )
        # If no boxes were merged, we're done
        if not merged_any:
            break

    return regions.quads, regions.scores
