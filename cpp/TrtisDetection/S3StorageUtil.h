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

#ifndef TRTIS_DETECTION_S3STORAGEUTIL_H
#define TRTIS_DETECTION_S3STORAGEUTIL_H

#include <map>
#include <string>

#include <log4cxx/logger.h>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "MPFDetectionComponent.h"

using namespace std;
using namespace MPF::COMPONENT;

// TODO: Consider making part of the C++ Batch Component SDK.
class S3StorageUtil {
private:
    log4cxx::LoggerPtr _log;          ///< log object
    string s3_bucket_url;             ///< AWS S3 bucket url to use for job (e.g. 'http://localhost:80/bucket')
    string s3_endpoint;               ///< AWS S3 endpoint url to use for job (e.g. 'http://localhost:80')
    string s3_bucket;                 ///< AWS S3 bucket to use for job (e.g. 'bucket')
    Aws::SDKOptions aws_sdk_options;  ///< AWS S3 options
    unique_ptr <Aws::S3::S3Client> s3_client;  ///< AWS S3 client

    Aws::S3::Model::GetObjectOutcome _getS3Object(const string bucket_name,
                                                  const string &object_name) const;  ///< GetObjectOutcome for S3 object

public:
    S3StorageUtil(const log4cxx::LoggerPtr &log,
                  const MPFJob &job);

    S3StorageUtil(const log4cxx::LoggerPtr &log,
                  const string &resultsBucketUrl,
                  const string &accessKey,
                  const string &secretKey);

    virtual ~S3StorageUtil();

    static bool RequiresS3Storage(const MPFJob &job);                         ///< determine if AWS S3 storage is required; throws if missing/invalid properties

    static bool RequiresS3Storage(const string &resultsBucketUrl,
                                  const string &accessKey,
                                  const string &secretKey);                   ///< determine if AWS S3 storage is required; throws if missing/invalid properties

    string GetS3ResultsBucketUrl();                                           ///< get the AWS S3 results bucket URL (e.g. 'http://localhost:80/bucket')

    string GetS3ResultsEndpoint();                                            ///< get the AWS S3 results endpoint URL (e.g. 'http://localhost:80')

    string GetS3ResultsBucket();                                              ///< get the AWS S3 results bucket (e.g. 'bucket')

    static string GetSha256(const string &buffer);                            ///< cal sha256 for a string buffer

    static string GetObjectName(const string &hash);                          ///< get the object name in the form "xx/yy/hash"

    string PutS3Object(const string                  &buffer,
                       const std::map<string,string> &metaData = {}) const;   ///< write contents of string buffer out to S3 object

    string PutS3Object(const string                  &bucket_name,
                       const string                  &buffer,
                       const std::map<string,string> &metaData = {}) const;   ///< write contents of string buffer out to S3 object

    void GetS3Object(const string         &object_name,
                     string               &buffer) const;                     ///< read content of a object to a string buffer

    void GetS3Object(const string         &object_name,
                     string               &buffer,
                     map<string,string>   &metaData) const;                   ///< read content of a object to a string buffer

    void GetS3Object(const string         &bucket_name,
                     const string         &object_name,
                     string               &buffer) const;                     ///< read content of a object to a string buffer

    void GetS3Object(const string         &bucket_name,
                     const string         &object_name,
                     string               &buffer,
                     map<string,string>   &metaData) const;                   ///< read content of a object to a string buffer

    void DeleteS3Object(const string &bucket_name,
                        const string &object_name) const;                     ///< delete an object from an S3 bucket

    void DeleteS3Object(const string &object_name) const;                     ///< delete an object from an S3 bucket

    bool ExistsS3Object(const string &object_name) const;                     ///< check if an object exists in an S3 bucket

    bool ExistsS3Object(const string &bucket_name,
                        const string &object_name) const;                     ///< check if an object exists in an S3 bucket

    bool ExistsS3Bucket(const string &bucket_name="") const;                  ///< check if an S3 bucket exists

    void CreateS3Bucket(const string &bucket_name="") const;                  ///< create an S3 bucket if it does not exist

    void DeleteS3Bucket(const string &bucket_name="") const;                  ///< delete an S3 bucket if it exists

    void EmptyS3Bucket(const string  &bucket_name="") const;                  ///< remove all objects from a bucket
};

#endif //TRTIS_DETECTION_S3STORAGEUTIL_H
