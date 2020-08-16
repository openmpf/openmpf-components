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

#ifndef TRTIS_DETECTION_S3STORAGEHELPER_H
#define TRTIS_DETECTION_S3STORAGEHELPER_H

#include <map>
#include <string>

#include <log4cxx/logger.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "MPFDetectionComponent.h"
#include "MPFVideoCapture.h"

using namespace std;
using namespace MPF::COMPONENT;

class S3StorageHelper {
  private:
    log4cxx::LoggerPtr _log;          ///< log object
    string s3_bucket;                 ///< amazon web services s3 bucket to use for job e.g. 'bucket'
    string s3_bucket_url;             ///< amazon web services s3 bucket url to use for job e.g. 'http://localhost:80/bucket'
    Aws::S3::S3Client s3_client;      ///< amazon web services s3 client
    Aws::SDKOptions aws_sdk_options;  ///< amazon web services s3 options

  public:
    S3StorageHelper() = default;
    S3StorageHelper(const log4cxx::LoggerPtr &log,
                    const string &resultsBucketUrl,
                    const string &accessKey,
                    const string &secretKey);

    ~S3StorageHelper();

    bool IsValid();                                                           ///< check if this helper is valid

    string GetSha256(const string &buffer);                                   ///< cal sha256 for a string buffer

    map<string,string> PrepS3Meta(const string     &data_uri,
                                  const string     &model,
                                  MPFImageLocation &loc);                     ///< collection map to use as s3 meta-data

    map<string,string> PrepS3Meta(const string   &data_uri,
                                   const string   &model,
                                   MPFVideoTrack  &track,
                                   const double   &fps=0.0);                  ///< collection map to use as s3 meta-data

    MPFVideoTrack DummyTransform(const MPFVideoCapture &video_cap,
                                 const int              frame_idx,
                                 const MPFImageLocation loc);                 ///< copy a location to dummy track and reverse transfrom

    void AddToTrack(MPFImageLocation &location,
                     int               frame_index,
                     MPFVideoTrack     &track);                               ///< add location to a track

    string PutS3Object(const string                  &buffer,
                       const std::map<string,string> &metaData = {});         ///< write contents of string buffer out to s3 object

    bool GetS3Object(const string         &object_name,
                     string               &buffer);                           ///< read content of a object to a string buffer

    bool GetS3Object(const string         &object_name,
                     string               &buffer,
                     map<string,string>   &metaData);                         ///< read content of a object to a string buffer

    bool DeleteS3Object(const string &object_name);                           ///< delete an object from an S3 bucket

    bool ExistsS3Object(const string &object_name);                           ///< check if an object exists in an S3 bucket

    bool ExistsS3Bucket(const string &bucket_name="");                        ///< check if an S3 bucket exists

    bool CreateS3Bucket(const string &bucket_name="");                        ///< create an S3 bucket if it does not exist

    bool DeleteS3Bucket(const string &bucket_name="");                        ///< delete an S3 bucket if it exists

    bool EmptyS3Bucket(const string  &bucket_name="");                        ///< remove all objects from a bucket
};

#endif //TRTIS_DETECTION_S3STORAGEHELPER_H
