/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

#include <cstdio>
#include <opencv2/core.hpp>

#include "MPFDetectionComponent.h"
#include "Utils.h"

#include "DetectionComparisonA.h"

using std::vector;

namespace MPF { namespace COMPONENT { namespace DetectionComparisonA {

    namespace {
        bool CompareDetections(const MPFImageLocation &query_detection, const MPFImageLocation &target_detection, bool log) {
            cv::Rect target_detection_rect = Utils::ImageLocationToCvRect(target_detection);
            cv::Rect query_detection_rect = Utils::ImageLocationToCvRect(query_detection);
            cv::Rect intersection = target_detection_rect & query_detection_rect;
            int intersection_area = intersection.area();
            double target_intersection_area = floor(target_detection_rect.area()) * 0.1;
            bool same = intersection_area > target_intersection_area;
            if (log && !same) {
                printf("\tCalc intersection < min target: %d < %f\n", intersection_area, target_intersection_area);
            }
            return same;
        }

        int CompareTracks(const MPFVideoTrack &query_track, const MPFVideoTrack &target_track) {
            int query_track_start_frame = query_track.start_frame;
            int target_track_start_frame = target_track.start_frame;

            int query_track_stop_frame = query_track.stop_frame;
            int target_track_stop_frame = target_track.stop_frame;

            int loop_start_index = 0;
            int query_track_index_modifier = target_track_start_frame - query_track_start_frame;
            if (query_track_index_modifier < 0) {
                loop_start_index = abs(query_track_index_modifier);
            }

            int loop_end_count = -1;
            if (query_track_stop_frame < target_track_stop_frame) {
                loop_end_count = query_track_stop_frame - target_track_start_frame;
            }
            else {
                loop_end_count = static_cast<int>(target_track.frame_locations.size());
            }

            int matched_detections = 0;
            for (int k = loop_start_index; k < loop_end_count; k++) {
                const MPFImageLocation &target_track_detection =
                    target_track.frame_locations.at(target_track_start_frame + k);
                auto qtdItr = query_track.frame_locations.find(query_track_start_frame + k + query_track_index_modifier);
                if(qtdItr != query_track.frame_locations.end()){
                  const MPFImageLocation &query_track_detection = query_track.frame_locations.at(query_track_start_frame + k + query_track_index_modifier);
                  if (CompareDetections(query_track_detection, target_track_detection, true)) {
                      matched_detections++;
                  }
                }
            }

            return matched_detections;
        }

        int FindTrack(const MPFVideoTrack &known_track, const vector<MPFVideoTrack> &actual_tracks, int frame_diff) {
            const MPFImageLocation &first_known_detection = known_track.frame_locations.begin()->second;

            for (unsigned int i = 0; i < actual_tracks.size(); i++) {
                const MPFVideoTrack &actual_track = actual_tracks.at(i);

                if (abs(known_track.start_frame - actual_track.start_frame) == frame_diff) {
                    for (const auto &actual_pair : actual_track.frame_locations) {

                        // Weak track match: Only one detection between the tracks needs to overlap.
                        if (CompareDetections(actual_pair.second, first_known_detection, false)) {
                            return i;
                        }
                    }
                }
            }

            return -1;
        }

        int FindTrack(const MPFVideoTrack &known_track, const vector<MPFVideoTrack> &actual_tracks) {
            for (int i = 0; i < 5; i++) {
                int match_track_index = FindTrack(known_track, actual_tracks, i);
                if (match_track_index != -1) {
                    return match_track_index;
                }
            }
            return -1;
        }
    } // anonymous namespace

    float CompareDetectionOutput(const vector<MPFVideoTrack> &actual_tracks,
                                 const vector<MPFVideoTrack> &known_tracks) {
        float total_score = 0.0;
        int total_known_detections = 0;

        for (unsigned int i = 0; i < known_tracks.size(); ++i) {
            total_known_detections += static_cast<int>(known_tracks.at(i).frame_locations.size());
        }

        int matched_detections = 0;
        int total_actual_detections = 0;
        for (unsigned int i = 0; i < actual_tracks.size(); ++i) {
            total_actual_detections += static_cast<int>(actual_tracks.at(i).frame_locations.size());
        }

        float track_count_factor = 1.0;
        if (total_actual_detections > total_known_detections) {
            printf("There are more actual detections than expected detections: ");
            track_count_factor = fabsf(static_cast<float>(total_known_detections) / static_cast<float>(total_actual_detections));
        }
        else if (total_actual_detections < total_known_detections) {
            printf("There are less actual detections than expected detections: ");
            track_count_factor = fabsf(static_cast<float>(total_actual_detections) / static_cast<float>(total_known_detections));
        }
        else {
            printf("Same number of actual and expected detections: ");
        }

        printf("%d actual vs. %d known\n", total_actual_detections, total_known_detections);

        vector<MPFVideoTrack> known_tracks_copy(known_tracks);
        vector<MPFVideoTrack> actual_tracks_copy(actual_tracks);
        while (!known_tracks_copy.empty()) {
            const MPFVideoTrack &known_track = known_tracks_copy.front();

            // Match the known track to as many actual tracks as possible. This is done to address the case where the
            // component generates multiple tracks instead of one, maybe due to non-determinism or an OpenCV upgrade.
            // These are weak track matches. Only one detection between the known track and actual track need to overlap
            // to match. Thus, the number of successfully matched detections can be significantly reduced due to weak
            // track matches. One effect is that even if the actual output is exactly the same as the known output,
            // this comparison approach can result in a score < 1.
            int match_track_index;
            do {
                match_track_index = FindTrack(known_track, actual_tracks_copy);
                if (match_track_index != -1) {
                    const MPFVideoTrack &match_track = actual_tracks_copy.at(match_track_index);
                    matched_detections += CompareTracks(match_track, known_track);
                    actual_tracks_copy.erase(actual_tracks_copy.begin() + match_track_index);
                    // This break will result in a 1-to-1 matching between known tracks and actual tracks. Uncommenting
                    // this line can enable a score == 1 that is otherwise not possible due to 1-to-many track matches.
                    // break;
                }
            } while(match_track_index != -1);

            known_tracks_copy.erase(known_tracks_copy.begin());
        }

        printf("\t\tMatched detections:\t\t%d\n", matched_detections);
        printf("\t\tTotal expected detections:\t%d\n", total_known_detections);
        printf("\t\tTrack count factor:\t\t%f\n", track_count_factor);
        printf("\t\tCombined:\t\t\t(%d/%d)*%f\n", matched_detections, total_known_detections, track_count_factor);

        total_score = (static_cast<float>(matched_detections) / static_cast<float>(total_known_detections)) * track_count_factor;
        printf("\t\tTotal score:\t\t\t%f\n", total_score);

        return total_score;
    }

    float CompareDetectionOutput(const vector<MPFImageLocation> &actual_detections,
                                 const vector<MPFImageLocation> &known_detections) {
        float total_score = 1.0;
        int total_actual_detections = static_cast<int>(actual_detections.size());
        int matched_detections = 0;
        int total_known_detections = static_cast<int>(known_detections.size());

        float track_count_factor = 1.0;
        if (total_actual_detections > total_known_detections) {
            printf("There are more actual detections than expected detections: ");
            track_count_factor = fabsf(static_cast<float>(total_known_detections) / static_cast<float>(total_actual_detections));
        }
        else if (total_actual_detections < total_known_detections) {
            printf("There are less actual detections than expected detections: ");
            track_count_factor = fabsf(static_cast<float>(total_actual_detections) / static_cast<float>(total_known_detections));
        }
        else {
            printf("Same number of actual and expected detections: ");
        }

        printf("%d actual vs. %d known\n", total_actual_detections, total_known_detections);
        float overlapTotal = 0.0;
        int overlapCount = 0;
        for (int i = 0; i < actual_detections.size(); i++) {
            float bestOverlap = 0.0;
            cv::Rect actual_detection_rect = Utils::ImageLocationToCvRect(actual_detections.at(i));

            for (int j = 0; j < known_detections.size(); j++) {
                cv::Rect known_detection_rect = Utils::ImageLocationToCvRect(known_detections.at(j));
                cv::Rect intersection = known_detection_rect & actual_detection_rect;
                float overlap = (intersection.area() > known_detection_rect.area()) ?
                    static_cast<float>(known_detection_rect.area()) / static_cast<float>(intersection.area()) :
                    static_cast<float>(intersection.area()) / static_cast<float>(known_detection_rect.area());
                bestOverlap = (overlap > bestOverlap) ? overlap : bestOverlap;
            }
            overlapTotal += bestOverlap;
            overlapCount++;
        }

        total_score = track_count_factor * (overlapTotal / static_cast<float>(overlapCount));
        printf("Total score: %f\n", total_score);
        return total_score;
    }
}}}
