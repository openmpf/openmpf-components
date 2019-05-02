import cv2
import numpy as np


contour_area = np.vectorize(
    lambda b: cv2.contourArea(b.astype(np.float32)),
    signature='(4,2)->()'
)
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



def nms_merge(quads, rotations, heights, areas, scores,
              overlap_threshold, text_height_threshold, rotation_threshold):
    order = np.argsort(areas)[::-1]
    quads = quads[order]
    rotations = rotations[order]
    heights = heights[order]
    scores = scores[order]
    areas = areas[order]

    merged_quads = []
    merged_rotations = []
    merged_heights = []
    merged_scores = []
    while len(quads) > 0:
        inter = contour_intersect_area(
            quads[0].astype(np.float32),
            quads[1:].astype(np.float32)
        )
        union = areas[0] + areas[1:] - inter
        # iou = inter / union
        iou = inter / areas[1:]
        to_merge = iou > overlap_threshold

        h_dist = 2 * np.abs(heights[0]-heights[1:]) / (heights[0]+heights[1:])
        height_matches = h_dist < text_height_threshold
        to_merge = np.logical_and(to_merge, height_matches)

        r_dist = np.abs(rotations[0] - rotations[1:])
        rotation_matches = r_dist < (rotation_threshold * np.pi / 180)
        to_merge = np.logical_and(to_merge, rotation_matches)

        if not np.any(to_merge):
            merged_quads.append(quads[0])
            merged_rotations.append(rotations[0])
            merged_heights.append(heights[0])
            merged_scores.append(scores[0])
            quads = quads[1:]
            rotations = rotations[1:]
            heights = heights[1:]
            areas = areas[1:]
            scores = scores[1:]
            continue

        to_merge = np.hstack((True, to_merge))

        weights = scores[to_merge]
        scale = 1.0 / weights.sum()

        merged_rotation = np.sum(rotations[to_merge] * weights) * scale
        merged_quad = merge(quads[to_merge], merged_rotation)

        merged_quads.append(merged_quad)
        merged_rotations.append(merged_rotation)
        merged_heights.append(np.sum(heights[to_merge] * weights) * scale)
        merged_scores.append(np.sum(scores[to_merge] * weights) * scale)

        unmerged = np.logical_not(to_merge)

        quads = quads[unmerged]
        rotations = rotations[unmerged]
        heights = heights[unmerged]
        areas = areas[unmerged]
        scores = scores[unmerged]

    quads = np.stack(merged_quads)
    areas = contour_area(quads.astype(np.float32))
    return dict(
        quads=quads,
        rotations=np.array(merged_rotations),
        heights=np.array(merged_heights),
        areas=areas,
        scores=np.array(merged_scores)
    )

def lanms_approx(rboxes, scores, overlap_threshold, text_height_threshold, rotation_threshold):
    quads = rbox_to_quad(rboxes)
    kwargs = dict(
        quads=quads,
        rotations=rboxes[:,6],
        heights=rboxes[:,2] + rboxes[:,4],
        areas=contour_area(quads),
        scores=scores
    )
    for i in range(30):
        kwargs = nms_merge(
            overlap_threshold=overlap_threshold,
            text_height_threshold=text_height_threshold,
            rotation_threshold=rotation_threshold,
            **kwargs
        )
    return kwargs['quads'], kwargs['scores']
