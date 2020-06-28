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

#ifndef OCVSSDFACEDETECTION_TRACK_H
#define OCVSSDFACEDETECTION_TRACK_H

#include <log4cxx/logger.h>

#include <opencv2/tracking.hpp>

#include "types.h"
#include "JobConfig.h"
#include "DetectionLocation.h"
#include "KFTracker.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    class Track{

      public:

        static bool Init(log4cxx::LoggerPtr log, const string plugin_path);  ///< setup class shared members

        DetectionLocationPtr ocvTrackerPredict(const JobConfig &cfg);        ///< predict a new detection from an exiting one using a tracker
        void releaseTracker() {_trackerPtr.release();}                       ///< release tracker so it can be reinitialized

        // Vector like interface detection pointer in tack
        const DetectionLocationPtr &at        (size_t i) const {return _locationPtrs.at(i);}
        const DetectionLocationPtr &operator[](size_t i) const {return _locationPtrs[i];}
        const DetectionLocationPtr &front()              const {return _locationPtrs.front();}
        const DetectionLocationPtr &back()               const {return _locationPtrs.back();}
        const size_t                         size()      const {return _locationPtrs.size();}
        DetectionLocationPtrVec::iterator    begin()           {return _locationPtrs.begin();}
        DetectionLocationPtrVec::iterator    end()             {return _locationPtrs.end();}
        void                                 push_back(DetectionLocationPtr d);

        void kalmanPredict(float t);
        void kalmanCorrect();
        #ifndef NDEBUG
        void kalmanDump(){_kfPtr->dump();};
        #endif

        Track(DetectionLocationPtr detPtr, const JobConfig &cfg);

      private:
        static log4cxx::LoggerPtr                _log;      ///< shared log object

        DetectionLocationPtrVec _locationPtrs;              ///< vector of pointers to locations  making up track
        cv::Ptr<cv::Tracker>    _trackerPtr;                ///< openCV tracker to help bridge gaps when detector fails
        size_t                  _trackerStartFrameIdx;      ///< frame index at which the tracker was initialized

        unique_ptr<KFTracker>   _kfPtr;                     ///< kalman filter tracker

    };

    /** **************************************************************************
    *   Dump MPF::COMPONENT::Track to a stream
    *************************************************************************** */
    inline
    ostream& operator<< (ostream& out, const Track& t) {
      out << "<f"   << t.front()->frameIdx << (MPFImageLocation)(*t.front())
          << "...f" << t.back()->frameIdx  << (MPFImageLocation)(*t.back())
          << ">("<<t.size()<<")";
      return out;
    }
  }
}

#endif