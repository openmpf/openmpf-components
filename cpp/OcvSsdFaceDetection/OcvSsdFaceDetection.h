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


#ifndef OPENMPF_COMPONENTS_OCVSsdFACEDETECTION_H
#define OPENMPF_COMPONENTS_OCVSsdFACEDETECTION_H

#include <log4cxx/logger.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

#include "types.h"
#include "Track.h"
#include "DetectionLocation.h"
#include "JobConfig.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;


  class OcvSsdFaceDetection : public MPFImageAndVideoDetectionComponentAdapter {

    public:
      bool Init()  override;
      bool Close() override;
      string GetDetectionType(){return "FACE";};
      MPFVideoTrackVec    GetDetections(const MPFVideoJob &job)    override;
      MPFImageLocationVec GetDetections(const MPFImageJob &job) override;

    private:

      log4cxx::LoggerPtr             _log;              ///< log object
      cv::Ptr<cv::CLAHE>             _equalizerPtr;     ///< adaptive histogram equalizer

      template<DetectionLocationCostFunc COST_FUNC>
      vector<long> _calcAssignmentVector(const TrackPtrList            &tracks,
                                         const DetectionLocationPtrVec &detections,
                                         const float                    maxCost); ///< determine costs of assigning detections to tracks

      MPFVideoTrack _convert_track(Track &track);  ///< convert to MFVideoTrack and release

      void _equalizeHistogram(JobConfig &cfg);     ///< perform histogram equalization on the frame
      void _normalizeFrame(JobConfig &cfg);        ///< perform image normalization

      void _assignDetections2Tracks(TrackPtrList            &tracks,
                                    DetectionLocationPtrVec &detections,
                                    const vector<long>      &assignmentVector,
                                    TrackPtrList            &assignedTracks);  ///< assign detections to tracks

  };
 }
}
#endif //OPENMPF_COMPONENTS_OCVFACEDETECTION_H
