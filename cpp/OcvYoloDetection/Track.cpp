/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

#include <cmath>
#include <vector>

#include "util.h"
#include "Track.h"

using namespace MPF::COMPONENT;


DetectionLocation &Track::front() {
    return locations_.front();
}


const DetectionLocation &Track::front() const {
    return locations_.front();
}


DetectionLocation &Track::back() {
    return locations_.back();
}


const DetectionLocation &Track::back() const {
    return locations_.back();
}


void Track::add(DetectionLocation detectionLocation) {
    if (!locations_.empty()) {
        // old tail's image no longer needed
        back().frame.data.release();
        assert(("Track frames have to be in sequence.",
                back().frame.idx < detectionLocation.frame.idx));
    }
    locations_.push_back(std::move(detectionLocation));
}


MPFVideoTrack Track::toMpfTrack(Track track) {
    assert(("Track start frame has to come before end frame.",
            track.front().frame.idx <= track.back().frame.idx));
    MPFVideoTrack mpfTrack(
            track.front().frame.idx,
            track.back().frame.idx,
            track.front().confidence);
    auto topClassItr = track.front().detection_properties.find("CLASSIFICATION");

    for (DetectionLocation &detection: track.locations_) {
        if (detection.confidence > mpfTrack.confidence) {
            mpfTrack.confidence = detection.confidence;
            topClassItr = detection.detection_properties.find("CLASSIFICATION");
        }
        assert(("All track frames have to fall between start and end frames.",
                track.front().frame.idx <= detection.frame.idx
                && detection.frame.idx <= track.back().frame.idx));
        mpfTrack.frame_locations.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(detection.frame.idx),
                std::forward_as_tuple(detection.x_left_upper, detection.y_left_upper,
                                      detection.width, detection.height, detection.confidence,
                                      std::move(detection.detection_properties))
        );
    }
    mpfTrack.detection_properties.insert(*topClassItr);
    return mpfTrack;
}


void Track::kalmanInit(const float t,
                       const float dt,
                       const cv::Rect2i &rec0,
                       const cv::Rect2i &roi,
                       const cv::Mat1f &rn,
                       const cv::Mat1f &qn) {
    kalmanFilterTracker_ = std::unique_ptr<KFTracker>(new KFTracker(t, dt, rec0, roi, rn, qn));
}


bool Track::ocvTrackerPredict(const Frame &frame, const long maxFrameGap, cv::Rect2i &prediction) {

    if (ocvTracker_.empty()) {   // initialize a new tracker if we don't have one already
        cv::Rect2i bbox = back().getRect();
        cv::Rect2i overlap = bbox & cv::Rect2i(0, 0, back().frame.data.cols - 1,
                                               back().frame.data.rows - 1);
        if (overlap.width > 1 && overlap.height > 1) {
            // could try different trackers here. e.g. cv::TrackerKCF::create();
            ocvTracker_ = cv::legacy::TrackerMOSSE::create();
            ocvTracker_->init(back().frame.data, bbox);
            LOG_TRACE("Tracker created for " << back());
            ocvTrackerStartFrameIdx_ = frame.idx;
        } else {
            LOG_TRACE("Can't create tracker for " << back());
            return false;
        }
    }

    if (frame.idx - ocvTrackerStartFrameIdx_ <= maxFrameGap) {
        cv::Rect2d pred;
        if (ocvTracker_->update(frame.data, pred)) {
            prediction.x = std::round(pred.x);
            prediction.y = std::round(pred.y);
            prediction.width = std::round(pred.width);
            prediction.height = std::round(pred.height);
            LOG_TRACE("Tracking " << back() << " to " << prediction);
            return true;
        } else {
            LOG_TRACE("Could not track " << back() << " to new location.");
        }
    }
    LOG_TRACE("Extrapolation tracking stopped" << back()
                                               << " frame gap = "
                                               << frame.idx - ocvTrackerStartFrameIdx_ << " > "
                                               << maxFrameGap);
    return false;
}


cv::Rect2i Track::predictedBox() const {
    if (kalmanFilterTracker_) {
        return kalmanFilterTracker_->predictedBBox();
    } else {
        return back().getRect();
    }
}

/** **************************************************************************
 * Advance Kalman filter state to predict next bbox at time t
 *
 * \param t time in sec to which Kalman filter state is advanced to
 *
*************************************************************************** */
void Track::kalmanPredict(const float t, const float edgeSnap) {
    if (kalmanFilterTracker_) {
        kalmanFilterTracker_->predict(t);
        // make frame edges "sticky"
        kalmanFilterTracker_->setStatePreFromBBox(
                snapToEdges(back().getRect(), kalmanFilterTracker_->predictedBBox(),
                            back().frame.data.size(), edgeSnap));
        LOG_TRACE("kf pred: " << back().getRect() << " => " << kalmanFilterTracker_->predictedBBox());
    }
}

/** **************************************************************************
 * apply Kalman correction to tail detection using tail's measurement
*************************************************************************** */
void Track::kalmanCorrect(const float edgeSnap) {
    if (kalmanFilterTracker_) {
        LOG_TRACE("kf meas: " << back().getRect());
        kalmanFilterTracker_->correct(back().getRect());
        cv::Rect2i corrected = snapToEdges(back().getRect(), kalmanFilterTracker_->correctedBBox(),
                                           back().frame.data.size(), edgeSnap);
        if ((corrected.width == 0) || (corrected.height == 0)) {
            kalmanFilterTracker_->setStatePostFromBBox(back().getRect());
        } else {
            kalmanFilterTracker_->setStatePostFromBBox(corrected);
            back().setRect(corrected);
        }
        LOG_TRACE("kf corr: " << back().getRect());
    }
}

float Track::testResidual(const cv::Rect2i &bbox, const float edgeSnap) const {
    if (kalmanFilterTracker_) {
        return kalmanFilterTracker_->testResidual(bbox, edgeSnap);
    } else {
        return 0.0;
    }
}


/** **************************************************************************
*   Dump MPF::COMPONENT::Track to a stream
*************************************************************************** */
std::ostream &operator<<(std::ostream &out, const Track &t) {
    out << "<f" << t.front().frame.idx << t.front()
        << "...f" << t.back().frame.idx << t.back()
        << ">(" << t.locations_.size() << ")";
    return out;
}
