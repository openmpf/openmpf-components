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

#include "MPFDetectionException.h"

#include "JsonFeatureStorage.h"
#include "base64.h"

using namespace MPF::COMPONENT;

void JsonFeatureStorage::_store(Properties &prop) {
  prop["FEATURE"] = Base64::Encode(prop.at("FEATURE")); // overwrite
}

void JsonFeatureStorage::Store(const string &data_uri,
                               const string &model,
                               MPFImageLocation &location) {
  _store(location.detection_properties);
}

void JsonFeatureStorage::Store(const string &data_uri,
                               const string &model,
                               const MPFVideoTrack &track,
                               MPFImageLocation &location,
                               double fp_ms) {
  _store(location.detection_properties);
}