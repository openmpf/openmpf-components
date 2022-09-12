/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
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

#include <string>
#include <vector>

#include <log4cxx/propertyconfigurator.h>
#include <MPFVideoCapture.h>

#include "Frame.h"
#include "Track.h"
#include "OcvYoloDetection.h"

#include "TestUtils.h"

using namespace std;
using namespace MPF::COMPONENT;

bool init_logging() {
    log4cxx::PropertyConfigurator::configure("log4cxx.properties");
    return true;
}


OcvYoloDetection initComponent() {
    OcvYoloDetection component;
    component.SetRunDirectory("../plugin");
    component.Init();
    return component;
}


float iou(const cv::Rect &r1, const cv::Rect &r2) {
    int intersectionArea = (r1 & r2).area();
    int unionArea = (r1 | r2).area();
    return static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}


float iou(const MPFImageLocation &l1, const MPFImageLocation &l2) {
    return iou(cv::Rect2i(l1.x_left_upper, l1.y_left_upper, l1.width, l1.height),
               cv::Rect2i(l2.x_left_upper, l2.y_left_upper, l2.width, l2.height));
}


Properties getTinyYoloConfig(float confidenceThreshold) {
    return {
            {"MODEL_NAME",           "tiny yolo"},
            {"NET_INPUT_IMAGE_SIZE", "416"},
            {"CONFIDENCE_THRESHOLD", std::to_string(confidenceThreshold)},
            {"FRAME_QUEUE_CAPACITY", "16"}
    };
}


Properties getYoloConfig(float confidenceThreshold) {
    return {
            {"MODEL_NAME",           "yolo"},
            {"NET_INPUT_IMAGE_SIZE", "416"},
            {"CONFIDENCE_THRESHOLD", std::to_string(confidenceThreshold)},
            {"FRAME_QUEUE_CAPACITY", "16"}
    };
}


Properties getTritonYoloConfig(const std::string &tritonServer, float confidenceThreshold) {
    return {
            {"MODEL_NAME",                   "yolo"},
            {"NET_INPUT_IMAGE_SIZE",         "608"},
            {"CONFIDENCE_THRESHOLD",         std::to_string(confidenceThreshold)},
            {"CUDA_DEVICE_ID",               "-1"},
            {"TRACKING_MAX_FRAME_GAP",       "10"},
            {"ENABLE_TRITON",                "true"},
            {"DETECTION_FRAME_BATCH_SIZE",   "16"},
            {"TRITON_SERVER",                tritonServer},
            {"TRITON_USE_SHM",               "false"}, // allow for remote server via plain gRPC
            {"TRITON_MAX_INFER_CONCURRENCY", "4"},
            {"FRAME_QUEUE_CAPACITY",         "16"}
    };
}


bool same(MPFImageLocation &l1, MPFImageLocation &l2,
          float confidenceTolerance, float iouTolerance,
          float &confidenceDiff,
          float &iouValue) {

    confidenceDiff = fabs(l1.confidence - l2.confidence);
    iouValue = iou(l1, l2);
    return (confidenceDiff <= confidenceTolerance)
           && (l1.detection_properties.at("CLASSIFICATION") == l1.detection_properties.at("CLASSIFICATION"))
           && (1.0f - iouValue <= iouTolerance);
}


bool same(MPFImageLocation &l1, MPFImageLocation &l2,
          float confidenceTolerance, float iouTolerance) {
    float tmp1, tmp2;
    return same(l1, l2, confidenceTolerance, iouTolerance, tmp1, tmp2);
}


bool same(MPFVideoTrack &t1, MPFVideoTrack &t2,
          float confidenceTolerance, float iouTolerance,
          float &confidenceDiff,
          float &aveIou) {

    confidenceDiff = fabs(t1.confidence - t2.confidence);
    if (t1.detection_properties.at("CLASSIFICATION") != t2.detection_properties.at("CLASSIFICATION")
        || confidenceDiff > confidenceTolerance) {
        return false;
    }

    int start_frame = min(t1.start_frame, t2.start_frame);
    int stop_frame = max(t1.stop_frame, t2.stop_frame);
    aveIou = 0.0;
    int numCommonFrames = 0;
    for (int f = start_frame; f <= stop_frame; ++f) {
        auto l1Itr = t1.frame_locations.find(f);
        auto l2Itr = t2.frame_locations.find(f);
        if (l1Itr != t1.frame_locations.end()
            && l2Itr != t2.frame_locations.end()) {
            aveIou += iou(l1Itr->second, l2Itr->second);
            ++numCommonFrames;
        } else if (l1Itr != t1.frame_locations.end()
                   || l2Itr != t2.frame_locations.end()) {
            ++numCommonFrames;
        }
    }
    aveIou /= numCommonFrames;
    return (1.0 - aveIou <= iouTolerance);
}


bool same(MPFVideoTrack &t1, MPFVideoTrack &t2, float confidenceTolerance, float iouTolerance) {
    float tmp1, tmp2;
    return same(t1, t2, confidenceTolerance, iouTolerance, tmp1, tmp2);
}


void write_track_output(vector<MPFVideoTrack> &tracks, const string& outTrackFileName, MPFVideoJob &videoJob) {
    std::ofstream outTrackFile(outTrackFileName);
    if (outTrackFile.is_open()) {
        for (int i = 0; i < tracks.size(); ++i) {
            outTrackFile << "#" << i << " ";
            outTrackFile << tracks.at(i) << std::endl;
        }
        outTrackFile.close();
    } else {
        std::cerr << "Could not open '" << outTrackFileName << "'" << std::endl;
        throw exception();
    }
}


vector<MPFVideoTrack> read_track_output(const string& inTrackFileName) {
    std::ifstream inTrackFile(inTrackFileName);
    vector<MPFVideoTrack> tracks;

    if (inTrackFile.is_open()) {
        int idx;
        inTrackFile.ignore(1000, '#');
        while (!inTrackFile.eof()) {
            inTrackFile >> idx;
            MPFVideoTrack track;
            inTrackFile >> track;
            tracks.insert(tracks.begin() + idx, track);
            inTrackFile.ignore(1000, '#');
        }
    } else {
        std::cerr << "Could not open '" << inTrackFileName << "'" << std::endl;
        throw exception();
    }
    return tracks;
}


void write_track_output_video(const string& inVideoFileName, vector<MPFVideoTrack> &tracks,
                              const string& outVideoFileName, MPFVideoJob &videoJob) {

    MPFVideoCapture cap(inVideoFileName);
    cv::VideoWriter writer(outVideoFileName,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           cap.GetFrameRate(), cap.GetFrameSize());
    // Sort tracks into frames
    map<int, vector<MPFVideoTrack *>> frameTracks;
    int trackIdx = 0;
    for (auto &track: tracks) {
        track.detection_properties.emplace("idx", std::to_string(trackIdx++));
        for (auto &det: track.frame_locations) {
            if (det.first < track.start_frame || det.first > track.stop_frame) {
                GOUT("\tdetection index " + to_string(det.first) + " outside of track frame range [" +
                     to_string(track.start_frame) + "," + to_string(track.stop_frame) + "]");
            }
        }
        for (int fr = track.start_frame; fr <= track.stop_frame; fr++) {
            frameTracks[fr].push_back(&track);
        }
    }

    // Render tracks
    cv::Mat frame;
    int frameIdx = cap.GetCurrentFramePosition();
    int calFrameIdx = round(cap.GetCurrentTimeInMillis() * cap.GetFrameRate() / 1000.0);
    int numColors = 16;
    std::vector<cv::Scalar> randomPalette;
    for (int i = 0; i < numColors; ++i) {
        randomPalette.emplace_back(rand() % 255, rand() % 255, rand() % 255);
    }
    while (cap.Read(frame)) {
        if (frameIdx > videoJob.stop_frame) break;
        if (frameIdx >= videoJob.start_frame) {
            auto tracksItr = frameTracks.find(frameIdx);
            if (tracksItr != frameTracks.end()) {
                for (MPFVideoTrack *trackPtr: tracksItr->second) {
                    auto detItr = trackPtr->frame_locations.find(frameIdx);
                    if (detItr != trackPtr->frame_locations.end()) {
                        cv::Rect detection_rect(detItr->second.x_left_upper, detItr->second.y_left_upper,
                                                detItr->second.width, detItr->second.height);


                        cv::rectangle(frame, detection_rect,
                                      randomPalette.at(atoi(trackPtr->detection_properties["idx"].c_str()) % numColors),
                                      2);
                        std::stringstream ss;
                        ss << trackPtr->detection_properties["idx"] << ":"
                           << detItr->second.detection_properties["CLASSIFICATION"] << ":" << std::setprecision(3)
                           << detItr->second.confidence;
                        cv::putText(frame, ss.str(), detection_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                    cv::Scalar(0, 200, 200), 1);
                    }
                }
            }
            string disp = "# " + to_string(frameIdx) + ":" + to_string(calFrameIdx);
            putText(frame, disp, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
            writer.write(frame);
        }
        frameIdx = cap.GetCurrentFramePosition();
        calFrameIdx = round(cap.GetCurrentTimeInMillis() * cap.GetFrameRate() / 1000.0);
    }

}


bool objectFound(const std::string &expectedObjectName, const Properties &detectionProperties) {
    return expectedObjectName == detectionProperties.at("CLASSIFICATION");
}


bool objectFound(const std::string &expectedObjectName, int frameNumber,
                 const std::vector<MPFVideoTrack> &tracks) {

    return std::any_of(tracks.begin(), tracks.end(), [&](const MPFVideoTrack &track) {
        return frameNumber >= track.start_frame
               && frameNumber <= track.stop_frame
               && objectFound(expectedObjectName, track.detection_properties)
               && objectFound(expectedObjectName,
                              track.frame_locations.at(frameNumber).detection_properties);

    });
}


bool objectFound(const std::string &expected_object_name,
                 const std::vector<MPFImageLocation> &detections) {
    return std::any_of(detections.begin(), detections.end(), [&](const MPFImageLocation &loc) {
        return objectFound(expected_object_name, loc.detection_properties);
    });
}
