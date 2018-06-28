/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#include "Trackers.h"

#include <unordered_map>

#include <opencv2/core.hpp>


using namespace MPF::COMPONENT;

namespace {
    cv::Rect to_rect(const MPFImageLocation &img_loc) {
        return { img_loc.x_left_upper, img_loc.y_left_upper, img_loc.width, img_loc.height };
    }

    bool order_by_descending_first_then_by_ascending_second(const std::pair<float, std::string> &left,
                                                            const std::pair<float, std::string> &right) {
        // Swap order of left.first and right.first in first tie argument to get descending confidence
        // Use regular order of left.second and right.second to get ascending names
        return std::tie(right.first, left.second) < std::tie(left.first, right.second);
    }
}


MPFImageLocation TrackingHelpers::CreateImageLocation(int num_classes_per_region, DarknetResult &detection) {
    auto &object_probs = detection.object_type_probs;
    int num_items_to_get = std::min(num_classes_per_region, static_cast<int>(object_probs.size()));
    auto last_item_iter = object_probs.begin() + num_items_to_get;

    // Put the first num_items_to_get items in sorted order.
    // Everything after last_item_iter, will be greater than *last_item_iter, but won't be in any specific order.
    // Using std::sort would take O(object_probs.size() * log(object_probs.size())) steps,
    // std::partial sort only takes O(object_probs.size() * log(num_items_to_get)) steps.
    std::partial_sort(object_probs.begin(), last_item_iter, object_probs.end(),
                      order_by_descending_first_then_by_ascending_second);

    auto prob_pair_iter = object_probs.begin();

    float top_confidence = prob_pair_iter->first;
    std::string top_confidence_class = prob_pair_iter->second;
    ++prob_pair_iter;

    std::ostringstream confidence_list;
    confidence_list << top_confidence;
    std::string classification_list = top_confidence_class;

    for (; prob_pair_iter != last_item_iter; ++prob_pair_iter) {
        confidence_list << "; " << prob_pair_iter->first;
        classification_list += "; " + prob_pair_iter->second;
    }

    const auto &rect = detection.detection_rect;

    return MPFImageLocation(rect.x, rect.y, rect.width, rect.height, top_confidence, {
            { "CLASSIFICATION", std::move(top_confidence_class) },
            { "CLASSIFICATION LIST", std::move(classification_list) },
            { "CLASSIFICATION CONFIDENCE LIST", confidence_list.str() }
    });
}


void TrackingHelpers::CombineImageLocations(const MPFImageLocation &new_img_loc, MPFImageLocation &existing_img_loc) {
    cv::Rect superset_region = to_rect(existing_img_loc) | to_rect(new_img_loc);

    existing_img_loc.x_left_upper = superset_region.x;
    existing_img_loc.y_left_upper = superset_region.y;
    existing_img_loc.width = superset_region.width;
    existing_img_loc.height = superset_region.height;
    // Calculate the probability that there was a detection in the existing image location OR a detection in detection_rect.
    // The probability of at least one of two independent non-mutually exclusive events can be calculated as:
    // P(A or B) = P(A) + P(B) - P(A) * P(B)
    float existing_confidence = existing_img_loc.confidence;
    float new_confidence = new_img_loc.confidence;
    existing_img_loc.confidence = existing_confidence + new_confidence - existing_confidence * new_confidence;
}




void PreprocessorTracker::ProcessFrameDetections(const std::vector<DarknetResult> &new_detections, int frame_number) {
    for (auto &darknet_result : new_detections) {
        auto &rect = darknet_result.detection_rect;
        for (const std::pair<float, std::string> &class_prob : darknet_result.object_type_probs) {
            BaseTracker::ProcessFrameDetections(
                    MPFImageLocation(rect.x, rect.y, rect.width, rect.height, class_prob.first,
                                     { {"CLASSIFICATION", class_prob.second} }),
                    frame_number);
        }
    }
}


bool PreprocessorTracker::IsSameTrack(const MPFImageLocation &new_loc, int frame_number,
                                      const MPFVideoTrack &existing_track) {
    return new_loc.detection_properties.at("CLASSIFICATION")
           == existing_track.detection_properties.at("CLASSIFICATION");
}

MPFVideoTrack PreprocessorTracker::CreateTrack(MPFImageLocation &&img_loc, int frame_number) {
    MPFVideoTrack track(frame_number, frame_number, img_loc.confidence,
                        { { "CLASSIFICATION", img_loc.detection_properties.at("CLASSIFICATION")} });
    track.frame_locations.emplace(frame_number, std::move(img_loc));
    return track;
}


void PreprocessorTracker::AddToTrack(MPFImageLocation &&new_img_loc, int frame_number,
                                     MPFVideoTrack &existing_track) {
    auto it = existing_track.frame_locations.find(frame_number);
    bool has_detection_in_current_frame = it != existing_track.frame_locations.end();
    if (has_detection_in_current_frame) {
        auto &existing_img_loc = it->second;
        TrackingHelpers::CombineImageLocations(new_img_loc, existing_img_loc);
        existing_track.confidence = std::max(existing_track.confidence, existing_img_loc.confidence);
    }
    else {
        existing_track.confidence = std::max(existing_track.confidence, new_img_loc.confidence);
        existing_track.frame_locations.emplace(frame_number, std::move(new_img_loc));
        existing_track.stop_frame = frame_number;
    }
}




DefaultTracker::DefaultTracker(int num_classes_per_region, double min_overlap)
    : RegionOverlapTracker(min_overlap)
    , num_classes_per_region_(num_classes_per_region)
{

}

bool DefaultTracker::OverlappingDetectionsAreSameTrack(const MPFImageLocation &new_loc, int frame_number,
                                                       const MPFVideoTrack &existing_track) {
    return new_loc.detection_properties.at("CLASSIFICATION")
           == existing_track.detection_properties.at("CLASSIFICATION");
}


MPFVideoTrack DefaultTracker::CreateTrack(MPFImageLocation &&img_loc, int frame_number) {
    MPFVideoTrack track(frame_number, frame_number, img_loc.confidence,
                        { { "CLASSIFICATION", img_loc.detection_properties.at("CLASSIFICATION") } });

    auto class_list_iter = img_loc.detection_properties.find("CLASSIFICATION LIST");
    if (class_list_iter != img_loc.detection_properties.end()) {
        track.detection_properties.insert(*class_list_iter);
    }
    auto confidence_list_iter = img_loc.detection_properties.find("CLASSIFICATION CONFIDENCE LIST");
    if (confidence_list_iter != img_loc.detection_properties.end()) {
        track.detection_properties.insert(*confidence_list_iter);
    }

    track.frame_locations.emplace(frame_number, std::move(img_loc));
    return track;
}


void DefaultTracker::AddToTrack(MPFImageLocation &&new_img_loc, int frame_number, MPFVideoTrack &existing_track) {
    auto same_frame_iter = existing_track.frame_locations.find(frame_number);
    bool current_frame_has_existing_detection = same_frame_iter != existing_track.frame_locations.end();
    if (current_frame_has_existing_detection) {
        auto &existing_img_loc = same_frame_iter->second;
        TrackingHelpers::CombineImageLocations(new_img_loc, existing_img_loc);
        existing_track.confidence = std::max(existing_track.confidence, existing_img_loc.confidence);
    }
    else {
        existing_track.confidence = std::max(existing_track.confidence, new_img_loc.confidence);
        existing_track.frame_locations.emplace(frame_number, std::move(new_img_loc));
        existing_track.stop_frame = frame_number;
    }
}

void DefaultTracker::ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number) {
    for (auto& darknet_result : new_detections) {
        RegionOverlapTracker::ProcessFrameDetections(
                TrackingHelpers::CreateImageLocation(num_classes_per_region_, darknet_result),
                frame_number);
    }
}

