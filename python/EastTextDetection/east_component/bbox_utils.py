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

def rbox_to_quad(rboxes):
    """ Convert RBOX geometry to bounding box corner coordinates
    RBOX format:
        [x, y, dt, dr, db, dl, rotation (rad)]
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]
    """
    # Get cosine and sine of the box angles
    cs = np.hstack((np.cos(rboxes[:,[6]]), np.sin(rboxes[:,[6]])))

    quads = np.empty((rboxes.shape[0],4,2),dtype=np.float32)

    # Translate to the left and right edges
    quads[:,[0,3],:] = (rboxes[:,[0,1]] + rboxes[:,[5]]*cs*[-1, +1])[:,None,:]
    quads[:,[1,2],:] = (rboxes[:,[0,1]] + rboxes[:,[3]]*cs*[+1, -1])[:,None,:]

    # Translate to the top and bottom edges
    quads[:,[0,1],:] -= (cs * rboxes[:,[2]])[:,None,::-1]
    quads[:,[2,3],:] += (cs * rboxes[:,[4]])[:,None,::-1]

    return quads

def quad_to_iloc(quads, scores):
    """ Convert QUAD geometry to OpenMPF ImageLocation format
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]
     OpenMPF ImageLocation format:
         [tl_x, tl_y, w, h, rotation (deg), score]
    """
    w_vec = quads[:,1] - quads[:,0]
    h_vec = quads[:,3] - quads[:,0]

    ilocs = np.empty((quads.shape[0],6), dtype=np.float32)
    ilocs[:,[0,1]] = quads[:,0]
    ilocs[:,2] = np.sqrt(np.sum(w_vec*w_vec, -1))
    ilocs[:,3] = np.sqrt(np.sum(h_vec*h_vec, -1))
    ilocs[:,4] = np.degrees(np.arctan2(-w_vec[:,1], w_vec[:,0])) % 360
    ilocs[:,5] = scores
    return ilocs

def quad_to_rbox(quads):
    """ Convert QUAD geometry to RBOX geometry
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]
     RBOX format:
         [x, y, dt, dr, db, dl, rotation (rad)]
    """
    w_vec = quads[:,1] - quads[:,0]
    h_vec = quads[:,3] - quads[:,0]

    rboxes = np.empty((quads.shape[0],7), dtype=np.float32)
    rboxes[:,[0,1]] = quads[:,0]
    rboxes[:,[2,5]] = 0
    rboxes[:,3] = np.sqrt(np.sum(w_vec*w_vec, -1))
    rboxes[:,4] = np.sqrt(np.sum(h_vec*h_vec, -1))
    rboxes[:,6] = np.arctan2(-w_vec[:,1], w_vec[:,0])
    return rboxes

def rbox_to_iloc(rboxes, scores):
    """ Convert RBOX geometry to OpenMPF ImageLocation format
     RBOX format:
         [x, y, dt, dr, db, dl, rotation (rad)]
     OpenMPF ImageLocation format:
         [tl_x, tl_y, w, h, rotation (deg), score]
    """
    c = np.cos(rboxes[:,6])
    s = np.sin(rboxes[:,6])

    ilocs = np.empty((rboxes.shape[0],6), dtype=np.float32)

    ilocs[:,0] = rboxes[:,0] - rboxes[:,5] * c - rboxes[:,2] * s
    ilocs[:,1] = rboxes[:,1] + rboxes[:,5] * s - rboxes[:,2] * c
    ilocs[:,2] = rboxes[:,3] + rboxes[:,5]
    ilocs[:,3] = rboxes[:,2] + rboxes[:,4]
    ilocs[:,4] = np.degrees(rboxes[:,6]) % 360
    ilocs[:,5] = scores
    return ilocs

def iloc_to_quad(ilocs):
    """ Convert RBOX geometry to bounding box corner coordinates
    OpenMPF ImageLocation format:
        [tl_x, tl_y, w, h, rotation (deg), score]
    QUAD format:
        [[tl_x, tl_y],
         [tr_x, tr_y],
         [br_x, br_y],
         [bl_x, bl_y]]
    """
    # Get cosine and sine of the box angles
    rotation = np.radians(ilocs[:,[4]])
    cs = np.hstack((np.cos(rotation), np.sin(rotation)))

    quads = np.empty((ilocs.shape[0],4,2),dtype=np.float32)

    # Set all to the top left
    quads[:] = ilocs[:,None,[0,1]]

    # Translate to the right side
    quads[:,[1,2],:] += (ilocs[:,[2]] * cs * [+1, -1])[:,None,:]

    # Translate to the bottom side
    quads[:,[2,3],:] += (cs * ilocs[:,[3]])[:,None,::-1]

    return quads

def nms(rboxes, scores, temp_padding, final_padding, min_nms_overlap):
    """ Perform Non-Maximum Suppression (NMS) on the given QUAD-formatted
        bounding boxes.
    """
    # Add temporary padding
    padded_rboxes = rboxes.copy()
    padded_rboxes[:,2:6] += temp_padding * (rboxes[:,[2]] + rboxes[:,[4]])

    # Perform NMS
    quads = rbox_to_quad(padded_rboxes)
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
        inds = np.flatnonzero(iou < min_nms_overlap)
        order = order[inds+1]

    # Filter the suppressed detections
    rboxes = rboxes[keep]
    scores = scores[keep]

    # Add final padding, convert to Image Loc
    rboxes[:,2:6] += final_padding * (rboxes[:,[2]] + rboxes[:,[4]])
    return rbox_to_iloc(rboxes, scores)

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

def close_under_modulus(a, b, abs_diff, modulus):
    """ Determines whether the absolute difference between a and b is less than
        abs_diff under modulus. For example, if a = -5, b = 13, and modulus =
        12, their difference is 6 under the modulus, so this function will
        return True when abs_diff >= 6, False otherwise.

        Functionally, this checks whether b falls in the modular range [a-
        abs_diff, a+abs_diff]. Note that if abs_diff >= modulus/2, the function
        will always return True.

        This function will work with numpy arrays, as long as a and b can be
        broadcasted together.
    """
    return (b + abs_diff - a) % modulus <= (2 * abs_diff)

class MergedRegions(object):
    def __init__(self, rboxes, scores):
        self.quads = rbox_to_quad(rboxes)
        self.rotations = rboxes[:,6]
        self.heights = rboxes[:,2] + rboxes[:,4]
        self.areas = contour_area(self.quads)
        self.weights = scores
        self.scores = scores
        self.constituents = [[i] for i in range(len(scores))]

    def perform_merge(self, constituent_index_lists):
        post_merge_quads = []
        post_merge_rotations = []
        post_merge_heights = []
        post_merge_weights = []
        post_merge_scores = []
        post_merge_constituents = []
        for indices in constituent_index_lists:
            if len(indices) == 1:
                i = indices[0]
                post_merge_quads.append(self.quads[i])
                post_merge_rotations.append(self.rotations[i])
                post_merge_heights.append(self.heights[i])
                post_merge_weights.append(self.weights[i])
                post_merge_scores.append(self.scores[i])
                post_merge_constituents.append(self.constituents[i])
                continue

            # The score for a merged region is the maximum constituent score
            score = self.scores[indices].max()

            # The weight for a merged region is the sum of constituent weights
            weights = self.weights[indices]
            scale = weights.sum()

            # The rotation and text height for a merged region is a weighted
            #  average of the constituent boxes' rotations and text heights
            rotation = np.sum(self.rotations[indices] * weights) / scale
            height = np.sum(self.heights[indices] * weights) / scale

            # Merge the constituent boxes at the weighted average rotation
            quad = merge(self.quads[indices], rotation)

            # Merge the constituent index lists
            constituents = [j for i in indices for j in self.constituents[i]]

            post_merge_quads.append(quad)
            post_merge_rotations.append(rotation)
            post_merge_heights.append(height)
            post_merge_weights.append(scale)
            post_merge_scores.append(score)
            post_merge_constituents.append(constituents)

        # Update members
        self.quads = np.stack(post_merge_quads)
        self.rotations = np.array(post_merge_rotations)
        self.heights = np.array(post_merge_heights)
        self.areas = contour_area(self.quads.astype(np.float32))
        self.weights = np.array(post_merge_weights)
        self.scores = np.array(post_merge_scores)
        self.constituents = post_merge_constituents

def merge_pass(regions, min_merge_overlap, max_height_delta, max_rot_delta):
    """ Complete one merge pass. This takes one box at a time (largest area
        first) and merges it with all eligible boxes. This merged box is added
        to a list, and all constituent boxes removed from the original list.
        The largest remaining box is then taken, and the process is repeated
        until the original list is empty.
    """
    # Sort by bounding box area (descending)
    order = np.argsort(regions.areas)[::-1]

    merged_any = False
    constituent_index_lists = []
    while len(order) > 0:
        head = order[0]
        order = order[1:]
        constituent_indices = [head]

        # If only one region remains, append and break
        if len(order) == 0:
            constituent_index_lists.append(constituent_indices)
            break

        # Filter out boxes without sufficient overlap. Based on intersection
        #  over smaller area (IoSA), not intersection over union (IoU). Because #  we select in order of area, the head region is always larger.
        inter = contour_intersect_area(
            regions.quads[head].astype(np.float32),
            regions.quads[order].astype(np.float32)
        )
        overlaps = (inter / regions.areas[order]) >= min_merge_overlap

        # Filter out boxes whose text height does not match
        diff = np.abs(regions.heights[head] - regions.heights[order])
        rel_diff = 2 * diff / (regions.heights[head] + regions.heights[order])
        height_matches = rel_diff <= max_height_delta

        # Filter out boxes whose rotation does not match
        rotation_matches = close_under_modulus(
            regions.rotations[head],
            regions.rotations[order],
            np.radians(max_rot_delta),
            180
        )

        # Merge the eligible boxes in the tail with the head box
        to_merge = np.all((overlaps, height_matches, rotation_matches), axis=0)
        if np.any(to_merge):
            constituent_indices.extend(order[to_merge])
            order = order[np.logical_not(to_merge)]
            merged_any = True

        constituent_index_lists.append(constituent_indices)

    # Update the regions structure in place
    if merged_any:
        regions.perform_merge(constituent_index_lists)

    return merged_any

def merge_regions(rboxes, scores, temp_padding, final_padding,
                  min_merge_overlap, max_height_delta, max_rot_delta):
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
        4. We threshold not the intersect over union, but the intersect over
           smallest area. This is because, as merged regions grow larger, the
           IoU between them and smaller boxes will shrink. Because the
           threshold value is constant, large boxes will eventually never merge
           with smaller boxes.
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


    # Add temporary padding
    padded_rboxes = rboxes.copy()
    padded_rboxes[:,2:6] += temp_padding * (rboxes[:,[2]] + rboxes[:,[4]])
    regions = MergedRegions(padded_rboxes, scores)

    # Do the initial locality-aware pass

    # Filter out boxes without sufficient overlap. Based on intersection
    #  over smaller area (IoSA), not intersection over union (IoU).
    inter = contour_intersect_area(
        regions.quads[:-1].astype(np.float32),
        regions.quads[1:].astype(np.float32)
    )
    smaller_area = np.minimum(regions.areas[:-1], regions.areas[1:])
    overlaps = (inter / smaller_area) >= min_merge_overlap

    # Filter out boxes whose text height does not match
    diff = np.abs(regions.heights[:-1] - regions.heights[1:])
    rel_diff = 2 * diff / (regions.heights[:-1] + regions.heights[1:])
    height_matches = rel_diff <= max_height_delta

    # Filter out boxes whose rotation does not match
    rotation_matches = close_under_modulus(
        regions.rotations[:-1],
        regions.rotations[1:],
        np.radians(max_rot_delta),
        180
    )

    # Merge the runs of eligible boxes
    to_merge = np.all((overlaps, height_matches, rotation_matches), axis=0)
    if np.any(to_merge):
        # Here, to_merge is the boolean map indicating whether the box at each
        #  index can be merged with the one directly after it.
        to_merge = np.hstack((to_merge, False))

        # Break down into runs of True cells. These runs can all be merged
        #  together at once. This is the approximation; with true locality
        #  aware NMS, boxes are merged successively, and the merging is done
        #  by averaging box corner coordinates. Here, we merge by taking a
        #  large box which encloses runs of overlapping constituents
        diffs = np.diff(np.hstack((False, to_merge)).astype(np.int8))
        run_starts = np.flatnonzero(diffs > 0)
        run_ends = np.flatnonzero(diffs < 0)

        # The first in a run of False cells is merged with the preceeding run
        to_merge[run_ends] = True

        constituent_index_lists = [
            list(range(a, b + 1))
            for a, b in zip(run_starts, run_ends)
        ]

        # Update the structure to include the merged runs as well as the
        #  original boxes left unmerged
        unmerged = np.flatnonzero(np.logical_not(to_merge))
        constituent_index_lists.extend([[i] for i in unmerged])
        regions.perform_merge(constituent_index_lists)

    # Continue until all eligible boxes have been merged
    while True:
        merged_any = merge_pass(
            regions,
            min_merge_overlap=min_merge_overlap,
            max_height_delta=max_height_delta,
            max_rot_delta=max_rot_delta
        )
        # If no boxes were merged, we're done
        if not merged_any:
            break

    # Merge the original unpadded boxes
    constituent_index_lists = regions.constituents
    regions = MergedRegions(rboxes, scores)
    regions.perform_merge(constituent_index_lists)

    # Convert to RBOX geometries, apply final padding, convert to Image Locs
    rboxes = quad_to_rbox(regions.quads)
    rboxes[:,2:6] += (final_padding * regions.heights)[:,None]
    return rbox_to_iloc(rboxes, regions.scores)
