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


#ifndef OPENMPF_COMPONENTS_OCVYOLODETECTION_H
#define OPENMPF_COMPONENTS_OCVYOLODETECTION_H

#include <log4cxx/logger.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

#include "types.h"
#include "Track.h"
#include "DetectionLocation.h"
#include "Config.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;


  class OcvYoloDetection : public MPFImageAndVideoDetectionComponentAdapter {

    public:
      bool Init()  override;
      bool Close() override;
      string GetDetectionType(){return "FACE";};
      MPFVideoTrackVec    GetDetections(const MPFVideoJob &job) override;
      MPFImageLocationVec GetDetections(const MPFImageJob &job) override;

    private:

      log4cxx::LoggerPtr _log;

      typedef float (DetectionLocation::*DetectionLocationCostFunc)(const Track &tr) const; ///< cost member-function pointer type

      template<DetectionLocationCostFunc COST_FUNC>
      vector<long> _calcAssignmentVector(const TrackList               &tracks,
                                         const DetectionLocationPtrVec &detections,
                                         const float                    maxCost,
                                         const float                    maxClassDist,
                                         const float                    maxKFResidual); ///< determine costs of assigning detections to tracks

      void _assignDetections2Tracks(TrackList               &tracks,
                                    DetectionLocationPtrVec &detections,
                                    const vector<long>      &assignmentVector,
                                    TrackList               &assignedTracks);          ///< assign detections to tracks

      MPFVideoTrack _convert_track(Track &track);  ///< convert to MFVideoTrack and release
  };
 }
}
#endif