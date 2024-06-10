/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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
#include <openssl/sha.h>

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/Delete.h>
#include <aws/s3/model/ObjectIdentifier.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/ListObjectVersionsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>

#include <detectionComponentUtils.h>
#include "MPFDetectionException.h"

#include "S3StorageUtil.h"
#include "uri.h"

S3StorageUtil::S3StorageUtil(const log4cxx::LoggerPtr &log,
                             const MPFJob &job) :
        S3StorageUtil(log,
                      DetectionComponentUtils::GetProperty(job.job_properties, "S3_RESULTS_BUCKET", ""),
                      DetectionComponentUtils::GetProperty(job.job_properties, "S3_ACCESS_KEY", ""),
                      DetectionComponentUtils::GetProperty(job.job_properties, "S3_SECRET_KEY", "")) {
}

S3StorageUtil::S3StorageUtil(const log4cxx::LoggerPtr &log,
                             const string &resultsBucketUrl,
                             const string &accessKey,
                             const string &secretKey) : _log(log) {
    if (resultsBucketUrl.empty()) {
        throw MPFDetectionException(MPF_MISSING_PROPERTY, "S3_RESULTS_BUCKET was not set.");
    }
    RequiresS3Storage(resultsBucketUrl, accessKey, secretKey); // will throw if access key or secret key missing

    if (_log->isTraceEnabled()) {
        aws_sdk_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
    } else {
        aws_sdk_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Off;
    }

    Aws::InitAPI(aws_sdk_options); // do this before creating the client configuration to prevent a segfault

    LOG4CXX_TRACE(_log, "Configuring S3 Client");
    Aws::Client::ClientConfiguration awsClientConfig;
    try {
        uri s3Url(resultsBucketUrl);
        s3_endpoint = s3Url.get_scheme() + "://" + s3Url.get_host();
        s3_endpoint += ((s3Url.get_port()) ? ":" + to_string(s3Url.get_port()) : "");
        awsClientConfig.endpointOverride = Aws::String(s3_endpoint.c_str());
        s3_bucket = s3Url.get_path();
        s3_bucket.erase(s3_bucket.find_last_not_of("/ ") + 1);
        s3_bucket_url = s3_endpoint + '/' + s3_bucket;
    } catch (exception ex) {
        throw MPFDetectionException(MPF_INVALID_PROPERTY, "Could not parse S3_RESULTS_BUCKET '" +
                                                          resultsBucketUrl + "': " + string(ex.what()));
    }

    Aws::Auth::AWSCredentials creds = Aws::Auth::AWSCredentials(Aws::String(accessKey.c_str()),
                                                                Aws::String(secretKey.c_str()));
    s3_client = unique_ptr<Aws::S3::S3Client>(new Aws::S3::S3Client(creds, awsClientConfig,
                                                                    Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                                                                    false));
}

S3StorageUtil::~S3StorageUtil() {
    Aws::ShutdownAPI(aws_sdk_options);
}

/** ****************************************************************************
* Determine if AWS S3 storage is required
*
* \param job  job with properties to check
*
* \returns true if AWS storage is required; false otherwise
***************************************************************************** */
bool S3StorageUtil::RequiresS3Storage(const MPFJob &job) {
    string resultsBucketUrl = DetectionComponentUtils::GetProperty(job.job_properties, "S3_RESULTS_BUCKET", "");
    string accessKey = DetectionComponentUtils::GetProperty(job.job_properties, "S3_ACCESS_KEY", "");
    string secretKey = DetectionComponentUtils::GetProperty(job.job_properties, "S3_SECRET_KEY", "");
    return RequiresS3Storage(resultsBucketUrl, accessKey, secretKey);
}

/** ****************************************************************************
* Determine if AWS S3 storage is required
*
* \param resultsBucketUrl  results bucket URL
* \param accessKey         access key
* \param secretKey         secret key
*
* \returns true if AWS storage is required; false otherwise
***************************************************************************** */
bool S3StorageUtil::RequiresS3Storage(const string &resultsBucketUrl,
                                      const string &accessKey,
                                      const string &secretKey) {
    if (resultsBucketUrl.empty()) {
        return false;
    }

    if (accessKey.empty() && secretKey.empty()) {
        throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                    "S3_RESULTS_BUCKET was set, but S3_ACCESS_KEY and S3_SECRET_KEY were not.");
    }

    if (accessKey.empty()) {
        throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                    "S3_RESULTS_BUCKET and S3_ACCESS_KEY were set, but S3_SECRET_KEY was not.");
    }

    if (secretKey.empty()) {
        throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                    "S3_RESULTS_BUCKET and S3_SECRET_KEY were set, but S3_ACCESS_KEY was not.");
    }

    return true;
}

string S3StorageUtil::GetS3ResultsBucketUrl() {
    return s3_bucket_url;
}

string S3StorageUtil::GetS3ResultsEndpoint() {
    return s3_endpoint;
}

string S3StorageUtil::GetS3ResultsBucket() {
    return s3_bucket;
}

/** ****************************************************************************
* Calculate the sha256 digest for a string buffer
*
* \param buffer  buffer to calculate the sha256 for
*
* \returns lowercase hexadecimal string corresponding to the sha256
***************************************************************************** */
string S3StorageUtil::GetSha256(const string &buffer) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, buffer.c_str(), buffer.length());
    SHA256_Final(hash, &sha256);
    ostringstream ss;
    ss << hex << setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << setw(2) << static_cast<unsigned>(hash[i]);
    }
    return ss.str();
}

/** ****************************************************************************
* Get the object name in the form "xx/yy/hash".
*
* \param hash  sha256 hash for the object
*
* \returns object name
***************************************************************************** */
string S3StorageUtil::GetObjectName(const string &hash) {
    string firstPair = hash.substr(0, 2);
    string secondPair = hash.substr(2, 2);
    return firstPair + '/' + secondPair + '/' + hash;
}

/** ****************************************************************************
* Write a string buffer to an S3 object in an S3 bucket
*
* \param object_name  give the name/key for the object
* \param buffer       the data to write
*
* \returns URL of object in S3 bucket
***************************************************************************** */
string S3StorageUtil::PutS3Object(const string &buffer,
                                  const map <string, string> &metaData) const {
    return PutS3Object(s3_bucket, buffer, metaData);
}

/** ****************************************************************************
* Write a string buffer to an S3 object in an S3 bucket
*
* \param bucket_name  give the name of the bucket
* \param object_name  give the name/key for the object in the bucket
* \param buffer       the data to write
*
* \returns URL of object in S3 bucket
***************************************************************************** */
string S3StorageUtil::PutS3Object(const string &bucket_name,
                                  const string &buffer,
                                  const map <string, string> &metaData) const {
    string objectKey = GetObjectName(GetSha256(buffer));
    Aws::S3::Model::PutObjectRequest req;
    req.SetBucket(bucket_name.c_str());
    req.SetKey(objectKey.c_str());
    for (auto &m : metaData) {
        req.AddMetadata(Aws::String(m.first.c_str(), m.first.size()),
                        Aws::String(m.second.c_str(), m.second.size()));
    }
    // see https://stackoverflow.com/questions/48666549/upload-uint8-t-buffer-to-aws-s3-without-going-via-filesystem
    auto data = Aws::MakeShared<Aws::StringStream>("PutObjectInputStream",
                                                   stringstream::in
                                                   | stringstream::out
                                                   | stringstream::binary);
    data->write(reinterpret_cast<char *>(const_cast<char *>(buffer.c_str())), buffer.length());
    req.SetBody(data);
    auto res = s3_client->PutObject(req);

    if (!res.IsSuccess()) {
        stringstream ss;
        ss << "Could not put object: " << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
        throw MPFDetectionException(MPF_FILE_WRITE_ERROR, ss.str());
    }

    return s3_endpoint + '/' + bucket_name + '/' + objectKey;
}

/** ****************************************************************************
* Read a string buffer to an S3 object in S3 bucket
*
* \param      object_name  give the name/key for the object in the bucket
* \param[out] buffer       where the data will be returned
***************************************************************************** */
void S3StorageUtil::GetS3Object(const string &object_name,
                                string &buffer) const {
    GetS3Object(s3_bucket, object_name, buffer);
}

/** ****************************************************************************
* Read a string buffer to an S3 object in S3 bucket
*
* \param      bucket_name  give the name of the bucket
* \param      object_name  give the name/key for the object in the bucket
* \param[out] buffer       where the data will be returned
***************************************************************************** */
void S3StorageUtil::GetS3Object(const string &bucket_name,
                                const string &object_name,
                                string &buffer) const {
    auto res = _getS3Object(bucket_name, object_name);

    // for alternative to string copy see https://github.com/aws/aws-sdk-cpp/issues/64
    auto &data = res.GetResultWithOwnership().GetBody();
    stringstream ss;
    ss << data.rdbuf();
    buffer = ss.str();
    LOG4CXX_TRACE(_log, "Retrieved '" << object_name << "' of size " << buffer.length());
}

/** ****************************************************************************
* Read a string buffer to an S3 object in an S3 bucket
*
* \param         object_name  give the name/key for the object in the bucket
* \param[out]    buffer       where the data will be returned
* \param[in,out] metadata     additional info about the object if desired
***************************************************************************** */
void S3StorageUtil::GetS3Object(const string &object_name,
                                string &buffer,
                                map <string, string> &metaData) const {
    GetS3Object(s3_bucket, object_name, buffer, metaData);
}

/** ****************************************************************************
* Read a string buffer to an S3 object in an S3 bucket
*
* \param         bucket_name   give the name of the bucket
* \param         object_name   give the name/key for the object in the bucket
* \param[out]    buffer        where the data will be returned
* \param[in,out] metadata      additional info  about the object if desired
***************************************************************************** */
void S3StorageUtil::GetS3Object(const string &bucket_name,
                                const string &object_name,
                                string &buffer,
                                map <string, string> &metaData) const {
    auto res = _getS3Object(bucket_name, object_name);

    // for alternative to string copy see https://github.com/aws/aws-sdk-cpp/issues/64
    auto &data = res.GetResultWithOwnership().GetBody();
    stringstream ss;
    ss << data.rdbuf();
    buffer = ss.str();
    LOG4CXX_TRACE(_log, "Retrieved '" << object_name << "' of size " << buffer.length());
    for (auto &p : res.GetResult().GetMetadata()) {
        metaData[string(p.first.c_str(), p.first.size())] = string(p.second.c_str(), p.second.size());
    }
}

Aws::S3::Model::GetObjectOutcome S3StorageUtil::_getS3Object(const string bucket_name,
                                                             const string &object_name) const {
    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(bucket_name.c_str());
    req.SetKey(object_name.c_str());
    Aws::S3::Model::GetObjectOutcome res = s3_client->GetObject(req);

    if (!res.IsSuccess()) {
        stringstream ss;
        ss << "Could not get object '" << object_name << "': "
           << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
        throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
    }

    return res;
}

/** ****************************************************************************
* Delete an object out of an S3 bucket
*
* \param object_name  give the name/key for the object in the bucket
***************************************************************************** */
void S3StorageUtil::DeleteS3Object(const string &object_name) const {
    DeleteS3Object(s3_bucket, object_name);
}

/** ****************************************************************************
* Delete an object out of an S3 bucket
*
* \param bucket_name  give the name of the bucket
* \param object_name  give the name/key for the object in the bucket
***************************************************************************** */
void S3StorageUtil::DeleteS3Object(const string &bucket_name,
                                   const string &object_name) const {
    Aws::S3::Model::DeleteObjectRequest req;
    req.SetBucket(bucket_name.c_str());
    req.SetKey(object_name.c_str());
    auto res = s3_client->DeleteObject(req);

    if (!res.IsSuccess()) {
        stringstream ss;
        ss << "Could not delete object '" << object_name << "': "
           << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
        throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
    }
}

/** ****************************************************************************
* Check if an object exists in an S3 bucket
*
* \param object_name  give the name/key for the object to check
*
* \returns true if successful
***************************************************************************** */
bool S3StorageUtil::ExistsS3Object(const string &object_name) const {
    return ExistsS3Object(s3_bucket, object_name);
}

/** ****************************************************************************
* Check if an object exists in an S3 bucket
*
* \param bucket_name  give the name of the bucket
* \param object_name  give the name/key for the object to check
*
* \returns true if successful
***************************************************************************** */
bool S3StorageUtil::ExistsS3Object(const string &bucket_name,
                                   const string &object_name) const {
    Aws::S3::Model::HeadObjectRequest req;
    req.SetBucket(bucket_name.c_str());
    req.SetKey(object_name.c_str());
    const auto res = s3_client->HeadObject(req);
    return res.IsSuccess();
}

/** ****************************************************************************
* Check if a bucket exists in an S3 store
*
* \param bucket_name  give the name of the bucket to check
*
* \returns true if successful
***************************************************************************** */
bool S3StorageUtil::ExistsS3Bucket(const string &bucket_name) const {
    Aws::S3::Model::HeadBucketRequest req;
    const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
    req.SetBucket(bucket.c_str());
    const auto res = s3_client->HeadBucket(req);
    if (res.IsSuccess()) {
        return true;
    }
    if (res.GetError().GetErrorType() == Aws::S3::S3Errors::RESOURCE_NOT_FOUND ||
        res.GetError().GetErrorType() == Aws::S3::S3Errors::NO_SUCH_BUCKET) {
        return false;
    }
    stringstream ss;
    ss << "Unable to determine if bucket '" << bucket << "' exists: "
       << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
    throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
}

/** ****************************************************************************
* Create a bucket in an S3 store if it does not exist
*
* \param bucket_name  give the name of the bucket to create
***************************************************************************** */
void S3StorageUtil::CreateS3Bucket(const string &bucket_name) const {
    const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
    if (ExistsS3Bucket(bucket)) {
        LOG4CXX_TRACE(_log, "No need to create bucket '" << bucket << "' because it already exists.");
        return;
    }

    Aws::S3::Model::CreateBucketRequest req;
    req.SetBucket(bucket.c_str());
    const auto res = s3_client->CreateBucket(req);

    if (!res.IsSuccess()) {
        stringstream ss;
        ss << "Unable to create bucket '" << bucket << "': "
           << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
        throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
    }
}

/** ****************************************************************************
* Delete a bucket in an S3 store if it exists
*
* \param bucket_name  give the name of the bucket to delete
***************************************************************************** */
void S3StorageUtil::DeleteS3Bucket(const string &bucket_name) const {
    const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
    if (!ExistsS3Bucket(bucket)) {
        LOG4CXX_TRACE(_log, "No need to delete bucket '" << bucket << "' because it does not exist.");
        return;
    }

    Aws::S3::Model::DeleteBucketRequest req;
    req.SetBucket(bucket.c_str());
    const auto res = s3_client->DeleteBucket(req);

    if (!res.IsSuccess()) {
        stringstream ss;
        ss << "Unable to delete bucket '" << bucket << "': "
           << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
        throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
    }
}

/** ****************************************************************************
* Empty a bucket in an S3 store if it exists
*
* \param bucket_name  give the name of the bucket to empty
***************************************************************************** */
void S3StorageUtil::EmptyS3Bucket(const string &bucket_name) const {
    const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
    if (!ExistsS3Bucket(bucket)) {
        LOG4CXX_TRACE(_log, "No need to empty bucket '" << bucket << "' because it does not exist.");
        return;
    }

    Aws::S3::Model::ListObjectsV2Request req;
    req.SetBucket(bucket.c_str());
    while (true) {
        auto res = s3_client->ListObjectsV2(req); // returns some or all (up to 1,000) of the objects in a bucket
        Aws::Vector <Aws::S3::Model::Object> oList = res.GetResult().GetContents();
        if (oList.empty()) {
            break;
        }

        Aws::S3::Model::Delete dObjs;
        // see https://github.com/aws/aws-sdk-cpp/issues/91
        for (auto const &o: oList) {
            Aws::S3::Model::ObjectIdentifier oId;
            oId.SetKey(o.GetKey());
            dObjs.AddObjects(oId);
        }
        Aws::S3::Model::DeleteObjectsRequest dReq;
        dReq.SetBucket(bucket.c_str());
        dReq.SetDelete(dObjs);
        auto dRes = s3_client->DeleteObjects(dReq);

        if (!res.IsSuccess()) {
            stringstream ss;
            ss << "Could delete all files in bucket '" << bucket << "': "
               << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
            throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, ss.str());
        }
    }
    // extra stuff needs to happen if versioned see https://docs.aws.amazon.com/AmazonS3/latest/dev/delete-or-empty-bucket.html#empty-bucket-awssdks
#ifdef VERSIONED_S3_OBJECTS
    // TODO: Complete me if and when necessary.
    Aws::S3::Model::ListObjectVersionsRequest req;
    req.SetBucket(bucket.c_str());
    auto res = cfg.s3_client.ListObjectVersions(req);
    Aws::Vector<Aws::S3::Model::Object> oList = res.GetResult().GetContents();
    while(true){
      for(auto const &o: oList){
        do stuff
      }
      if(!res.GetIsTruncated()){
        break;
      }
      res = cfg.s3_client.ListObjectsVersions(req);
    }
#endif
}