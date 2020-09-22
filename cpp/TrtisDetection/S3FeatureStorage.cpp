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

#include <iomanip>

#include <detectionComponentUtils.h>
#include "MPFDetectionException.h"

#include "S3FeatureStorage.h"

using namespace MPF::COMPONENT;

S3FeatureStorage::S3FeatureStorage(const MPFJob &job,
                                   const log4cxx::LoggerPtr &log) : s3StorageUtil(log, job) {
    // Fail fast if bucket does not exist.
    if (!s3StorageUtil.ExistsS3Bucket()) {
        throw MPFDetectionException(MPF_INVALID_PROPERTY,
                                    "S3_RESULTS_BUCKET '" + s3StorageUtil.GetS3ResultsBucketUrl() +
                                    "' does not exist.");
    }
}

map <string, string> S3FeatureStorage::_prepMeta(const string &data_uri,
                                                 const string &model,
                                                 const MPFImageLocation &location) {
    map <string, string> meta;

    const Properties &prop = location.detection_properties;
    meta.insert({"model", model});
    meta.insert({"data_uri", data_uri});
    meta.insert({"x", to_string(location.x_left_upper)});
    meta.insert({"y", to_string(location.y_left_upper)});
    meta.insert({"width", to_string(location.width)});
    meta.insert({"height", to_string(location.height)});

    meta.insert({"feature", prop.at("FEATURE TYPE")});
    if (prop.find("CLASSIFICATION") != prop.end()) {
        meta.insert({"class", prop.at("CLASSIFICATION")});
        if (location.confidence > 0) {
            stringstream conf;
            conf << setprecision(2) << fixed << location.confidence;
            meta.insert({"confidence", conf.str()});
        }
    }

    return meta;
}

map <string, string> S3FeatureStorage::_prepMeta(const string &data_uri,
                                                 const string &model,
                                                 const MPFVideoTrack &track,
                                                 const MPFImageLocation &location,
                                                 double fp_ms) {
    map <string, string> meta = _prepMeta(data_uri, model, location);

    meta.insert({"offsetFrame", to_string(track.start_frame)});
    if (fp_ms > 0.0) {
        stringstream ss;
        ss << setprecision(0) << fixed << track.start_frame / fp_ms;
        meta.insert({"offsetTime", ss.str()});
    }

    return meta;
}

void S3FeatureStorage::_store(const map <string, string> &meta,
                              Properties &prop) {
    prop["FEATURE URI"] = s3StorageUtil.PutS3Object(prop.at("FEATURE"), meta);
    prop.erase(prop.find("FEATURE"));
}

void S3FeatureStorage::Store(const string &data_uri,
                             const string &model,
                             MPFImageLocation &location) {
    map <string, string> meta = _prepMeta(data_uri, model, location);
    _store(meta, location.detection_properties);
}

void S3FeatureStorage::Store(const string &data_uri,
                             const string &model,
                             const MPFVideoTrack &track,
                             MPFImageLocation &location,
                             double fp_ms) {
    map <string, string> meta = _prepMeta(data_uri, model, track, location, fp_ms);
    _store(meta, location.detection_properties);
}