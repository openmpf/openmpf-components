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


#include <opencv2/core.hpp>

#include "Trackers.h"

using namespace MPF::COMPONENT;


namespace {
    bool order_by_descending_first_then_by_ascending_second(const std::pair<float, std::string> &left,
                                                            const std::pair<float, std::string> &right) {
        // Swap order of left.first and right.first in first tie argument to get descending confidence
        // Use regular order of left.second and right.second to get ascending names
        return std::tie(right.first, left.second) < std::tie(left.first, right.second);
    }
}


namespace TrackingHelpers {

    MPFImageLocation CreateImageLocation(int num_classes_per_region, DarknetResult &detection) {
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


    void CombineImageLocation(const cv::Rect &rect, float prob, MPFImageLocation &image_location) {
        cv::Rect existing_detection_rect(image_location.x_left_upper, image_location.y_left_upper,
                                         image_location.width, image_location.height);
        cv::Rect superset_region = rect | existing_detection_rect;

        image_location.x_left_upper = superset_region.x;
        image_location.y_left_upper = superset_region.y;
        image_location.width = superset_region.width;
        image_location.height = superset_region.height;
        // Calculate the probability that there was a detection in the existing image location OR a detection in detection_rect.
        // The probability of at least one of two independent non-mutually exclusive events can be calculated as:
        // P(A or B) = P(A) + P(B) - P(A) * P(B)
        image_location.confidence = image_location.confidence + prob - image_location.confidence * prob;
    }
}


DefaultTracker::DefaultTracker(int num_classes_per_region, double min_overlap)
        : num_classes_per_region_(num_classes_per_region)
        , min_overlap_(min_overlap)
{
}


void DefaultTracker::ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number) {
    for (DarknetResult &detection : new_detections) {
        MPFImageLocation img_loc = TrackingHelpers::CreateImageLocation(num_classes_per_region_, detection);
        std::string classification = img_loc.detection_properties.at("CLASSIFICATION");

        auto range = tracks_.equal_range({frame_number - 1, classification});
        double max_overlap = min_overlap_ - 1; // Can't set to -1.0 because min_overlap_ may be less than -1.
        auto max_overlap_iter = tracks_.end();
        for (auto it = range.first; it != range.second; ++it) {
            double overlap = GetOverlap(detection.detection_rect, it->second);
            if (overlap > max_overlap) {
                max_overlap = overlap;
                max_overlap_iter = it;
            }
        }

        if (max_overlap >= min_overlap_) {
            auto &track = max_overlap_iter->second;
            track.stop_frame = frame_number;
            track.confidence = std::max(track.confidence, img_loc.confidence);
            track.frame_locations.emplace(frame_number, std::move(img_loc));

            tracks_.emplace(std::make_pair(frame_number, classification), std::move(track));
            tracks_.erase(max_overlap_iter);
        }
        else {
            MPFVideoTrack track(frame_number, frame_number, img_loc.confidence, { {"CLASSIFICATION", classification}});
            track.frame_locations.emplace(frame_number, std::move(img_loc));
            tracks_.emplace(std::make_pair(frame_number, classification), std::move(track));
        }
    }
}

std::vector<MPFVideoTrack> DefaultTracker::GetTracks() {
    std::vector<MPFVideoTrack> results;
    results.reserve(tracks_.size());

    for (auto &pair : tracks_) {
        results.push_back(std::move(pair.second));
    }
    tracks_ = { };
    return results;
}


double DefaultTracker::GetOverlap(const cv::Rect &detection_rect, const MPFVideoTrack &track) {
    auto last_detection_it = track.frame_locations.rbegin();
    if (last_detection_it == track.frame_locations.rend()) {
        throw std::length_error("Track must not be empty.");
    }
    auto &last_detection = last_detection_it->second;
    cv::Rect track_rect(last_detection.x_left_upper, last_detection.y_left_upper,
                        last_detection.width, last_detection.height);

    if (track_rect.empty() || detection_rect.empty()) {
        return track_rect == detection_rect ? 1 : 0;
    }

    cv::Rect intersection = track_rect & detection_rect;
    cv::Rect rect_union = track_rect | detection_rect;
    return intersection.area() / static_cast<double>(rect_union.area());
}



void PreprocessorTracker::ProcessFrameDetections(const std::vector<DarknetResult> &new_detections, int frame_number) {
    for (const DarknetResult &location : new_detections) {
        for (const std::pair<float, std::string> &class_prob : location.object_type_probs) {
            // Check if there is more than one box in the current frame that has the same classification.
            auto current_frame_track_iter = tracks_.find({frame_number, class_prob.second});
            if (current_frame_track_iter != tracks_.cend()) {
                CombineImageLocation(location.detection_rect, class_prob.first, frame_number,
                                     current_frame_track_iter->second);
                continue;
            }
            // Check if the same type of object was found in the previous frame.
            auto previous_frame_track_iter = tracks_.find({frame_number - 1, class_prob.second});
            if (previous_frame_track_iter != tracks_.cend()) {
                AddNewImageLocationToTrack(location.detection_rect, class_prob.first, class_prob.second, frame_number,
                                           previous_frame_track_iter->second);
                continue;
            }
            AddNewTrack(location.detection_rect, class_prob.first, class_prob.second, frame_number);
        }
    }
}


void PreprocessorTracker::AddNewTrack(const cv::Rect &rect, float prob, const std::string &type,
                                      int frame_number) {

    MPFVideoTrack track(frame_number, frame_number, prob, { {"CLASSIFICATION", type} });
    track.frame_locations.emplace(
            frame_number,
            MPFImageLocation(rect.x, rect.y, rect.width, rect.height, prob, { {"CLASSIFICATION", type} }));

    tracks_.emplace(std::make_pair(frame_number, type), std::move(track));
}


void PreprocessorTracker::AddNewImageLocationToTrack(const cv::Rect &rect, float prob, const std::string &type,
                                                     int frame_number, MPFVideoTrack &track) {
    track.frame_locations.emplace(
            frame_number,
            MPFImageLocation(rect.x, rect.y, rect.width, rect.height, prob, Properties{ {"CLASSIFICATION", type} }));

    track.confidence = std::max(track.confidence, prob);
    track.stop_frame = frame_number;

    // Move track to its new key in tracks_;
    tracks_.emplace(std::make_pair(frame_number, type), std::move(track));
    tracks_.erase({frame_number - 1, type});
}


void PreprocessorTracker::CombineImageLocation(const cv::Rect &rect, float prob, int frame_number,
                                               MPFVideoTrack &track) {
    auto &frame_location = track.frame_locations.at(frame_number);
    TrackingHelpers::CombineImageLocation(rect, prob, frame_location);
    track.confidence = std::max(track.confidence, frame_location.confidence);
}


std::vector<MPF::COMPONENT::MPFVideoTrack> PreprocessorTracker::GetTracks() {
    std::vector<MPFVideoTrack> results;
    results.reserve(tracks_.size());

    for (auto &pair : tracks_) {
        results.push_back(std::move(pair.second));
    }
    tracks_.clear();
    return results;
}


size_t PreprocessorTracker::PairHasher::operator()(const std::pair<int, std::string> &pair) const {
    auto h1 = int_hasher_(pair.first);
    auto h2 = string_hasher_(pair.second);
    return h1 ^ (h2 << 1u);
}
