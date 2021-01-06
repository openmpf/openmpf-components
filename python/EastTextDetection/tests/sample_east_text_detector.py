#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
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
from __future__ import division, print_function

import sys
import os
import argparse
import mimetypes

import cv2

import mpf_component_api as mpf
import mpf_component_util as util

# Add east_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from east_component.east_component import EastComponent
from east_component.bbox_utils import *

# Context managers for video reading and writing
class ManagedVideoCapture(util.VideoCapture):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.release()

class ManagedVideoWriter(cv2.VideoWriter):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Test EAST component with one or files. Both video and images are '
            'supported. If --out-dir argument provided, marked-up images (but '
            'not video) will be saved in the specified directory.'
        )
    )
    parser.add_argument('--max-side-length', type=int, default=-1)
    parser.add_argument('--confidence-threshold', type=float, default=0.8)
    parser.add_argument('--merge-overlap-threshold', type=float, default=0.0)
    parser.add_argument('--nms-min-overlap', type=float, default=0.1)
    parser.add_argument('--merge-max-text-height-difference', type=float, default=0.3)
    parser.add_argument('--merge-max-rotation-difference', type=float, default=10.0)
    parser.add_argument('--min-structured-text-threshold', type=float, default=0.01)
    parser.add_argument('--rotate-and-detect', action='store_true')
    parser.add_argument('--merge-regions', action='store_true')
    parser.add_argument('--suppress-vertical', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temporary-padding-x', type=float, default=0.1)
    parser.add_argument('--temporary-padding-y', type=float, default=0.1)
    parser.add_argument('--final-padding', type=float, default=0.0)
    parser.add_argument(
        '--out-dir',
        nargs='?',
        help='directory to store marked-up images'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='locations of the image or video files'
    )
    args = parser.parse_args()

    markup = False
    if args.out_dir is not None:
        markup = True
        out_dir = os.path.realpath(args.out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    properties = dict(
        MAX_SIDE_LENGTH=str(args.max_side_length),
        CONFIDENCE_THRESHOLD=str(args.confidence_threshold),
        MERGE_OVERLAP_THRESHOLD=str(args.merge_overlap_threshold),
        NMS_MIN_OVERLAP=str(args.nms_min_overlap),
        MERGE_MAX_TEXT_HEIGHT_DIFFERENCE=str(args.merge_max_text_height_difference),
        MERGE_MAX_ROTATION_DIFFERENCE=str(args.merge_max_rotation_difference),
        MIN_STRUCTURED_TEXT_THRESHOLD=str(args.min_structured_text_threshold),
        ROTATE_AND_DETECT=str(args.rotate_and_detect),
        MERGE_REGIONS=str(args.merge_regions),
        SUPPRESS_VERTICAL=str(args.suppress_vertical),
        BATCH_SIZE=str(args.batch_size),
        TEMPORARY_PADDING_X=str(args.temporary_padding_x),
        TEMPORARY_PADDING_Y=str(args.temporary_padding_y),
        FINAL_PADDING=str(args.final_padding)
    )

    comp = EastComponent()

    for uri in args.files:
        filetype = mimetypes.guess_type(uri)[0].split('/')[0]
        print("Processing %s file: %s"%(filetype,uri))
        if filetype == 'image':
            job = mpf.ImageJob(
                job_name="EAST_test:%s"%uri,
                data_uri=uri,
                job_properties=properties,
                media_properties={},
                feed_forward_location=None
            )
            dets = {-1: list(comp.get_detections_from_image(job))}

            # If indicated, draw the detections and write to output directory
            if markup and len(dets[-1]):
                ttype = dets[-1][0].detection_properties['TEXT_TYPE']
                color = (0,255,0) if ttype == 'UNSTRUCTURED' else (255,0,255)
                ilocs = np.array([
                    [
                        float(d.x_left_upper),
                        float(d.y_left_upper),
                        float(d.width),
                        float(d.height),
                        float(d.detection_properties['ROTATION'])
                    ] for d in dets[-1]
                ])
                quads = iloc_to_quad(ilocs)
                frame = util.ImageReader(job).get_image().copy()
                cv2.drawContours(frame, np.int0(quads), -1, color, 2)
                outpath = os.path.join(out_dir, os.path.split(uri)[1])
                cv2.imwrite(outpath, frame)

        elif filetype == 'video':
            job = mpf.VideoJob(
                job_name='EAST_test',
                data_uri=uri,
                start_frame=0,
                stop_frame=-1,
                job_properties=properties,
                media_properties={},
                feed_forward_track=None
            )
            tracks = comp.get_detections_from_video(job)
            dets = {}
            for track in tracks:
                for fnum, iloc in track.frame_locations.items():
                    if fnum not in dets:
                        dets[fnum] = []
                    dets[fnum].append(iloc)

            if markup:
                with ManagedVideoCapture(job, False, False) as cap, ManagedVideoWriter(
                        os.path.join(out_dir, os.path.split(uri)[1]),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        cap.frame_rate,
                        cap.frame_size,
                        isColor=int(1)
                ) as writer:
                    for fnum, frame in enumerate(cap):
                        frame = frame.copy()
                        if fnum in dets:
                            ttype = dets[fnum][0].detection_properties['TEXT_TYPE']
                            color = (0,255,0) if ttype == 'UNSTRUCTURED' else (255,0,255)
                            ilocs = np.array([
                                [
                                    float(d.x_left_upper),
                                    float(d.y_left_upper),
                                    float(d.width),
                                    float(d.height),
                                    float(d.detection_properties['ROTATION'])
                                ] for d in dets[fnum]
                            ])
                            quads = np.int0(iloc_to_quad(ilocs))
                            cv2.drawContours(frame, quads, -1, color, 2)
                        writer.write(frame)
        else:
            print("Unreadable input file")
            continue

        for fnum, frame_dets in sorted(dets.items()):
            # If it's a video, print the frame number
            if fnum >= 0:
                print("   Frame %d"%fnum)

            print("   %d detections"%(len(frame_dets)))
            for j,d in enumerate(frame_dets):
                print((
                        "   Detection {:d}:\n"
                        "     Type:        {:s}\n"
                        "     Location:    ({:d}, {:d})\n"
                        "     Orientation: {:.3f} degrees\n"
                        "     Width:       {:d}\n"
                        "     Height:      {:d}\n"
                        "     Confidence:  {:.3%}\n"
                    ).format(
                        j,
                        d.detection_properties['TEXT_TYPE'],
                        int(d.x_left_upper),
                        int(d.y_left_upper),
                        float(d.detection_properties['ROTATION']),
                        int(d.width),
                        int(d.height),
                        float(d.confidence)
                    )
                )
