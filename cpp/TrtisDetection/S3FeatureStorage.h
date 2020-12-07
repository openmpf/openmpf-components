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

#ifndef TRTIS_DETECTION_S3FEATURESTORAGE_H
#define TRTIS_DETECTION_S3FEATURESTORAGE_H

#include "IFeatureStorage.h"
#include "S3StorageUtil.h"

using namespace std;
using namespace MPF::COMPONENT;

class S3FeatureStorage : public IFeatureStorage {
private:
    S3StorageUtil s3StorageUtil;

    map <string, string> _prepMeta(const string &data_uri,
                                   const string &model,
                                   const MPFImageLocation &location);

    map <string, string> _prepMeta(const string &data_uri,
                                   const string &model,
                                   const MPFVideoTrack &track,
                                   const MPFImageLocation &location,
                                   double fp_ms);

    void _store(const map <string, string> &meta,
                Properties &prop);

public:
    S3FeatureStorage(const MPFJob &job,
                     const log4cxx::LoggerPtr &log);

    void Store(const string &data_uri,
               const string &model,
               MPFImageLocation &loc) override;

    void Store(const string &data_uri,
               const string &model,
               const MPFVideoTrack &track,
               MPFImageLocation &location,
               double fp_ms) override;
};

#endif //TRTIS_DETECTION_S3FEATURESTORAGE_H
