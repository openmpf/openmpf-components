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

#include "MPFDetectionException.h"

#include "S3StorageHelper.h"
#include "uri.h"

S3StorageHelper::S3StorageHelper(const log4cxx::LoggerPtr &log,
                                 const string &resultsBucketUrl,
                                 const string &accessKey,
                                 const string &secretKey) : _log(log) {
  if (resultsBucketUrl.empty()) {
    throw MPFDetectionException(MPF_MISSING_PROPERTY, "Missing S3_RESULTS_BUCKET property.");
  }

  if (accessKey.empty() && secretKey.empty()) {
    throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                "The S3_RESULTS_BUCKET property was set, but the S3_ACCESS_KEY and S3_SECRET_KEY properties were not.");
  }

  if (accessKey.empty()) {
    throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                "The S3_RESULTS_BUCKET and S3_ACCESS_KEY properties were set, but the S3_SECRET_KEY property was not.");
  }

  if (secretKey.empty()) {
    throw MPFDetectionException(MPF_MISSING_PROPERTY,
                                "The S3_RESULTS_BUCKET and S3_SECRET_KEY properties were set, but the S3_ACCESS_KEY property was not.");
  }

  LOG4CXX_TRACE(_log, "Configuring S3 Client");
  Aws::Client::ClientConfiguration awsClientConfig;
  try {
    uri s3Url(resultsBucketUrl);
    string endpoint = s3Url.get_scheme() + "://" + s3Url.get_host();
    endpoint += ((s3Url.get_port()) ? ":" + to_string(s3Url.get_port()) : "");
    awsClientConfig.endpointOverride = Aws::String(endpoint.c_str());
    s3_bucket = s3Url.get_path();
    s3_bucket.erase(s3_bucket.find_last_not_of("/ ") + 1);
    s3_bucket_url = endpoint + '/' + s3_bucket;
  } catch (exception ex) {
    throw MPFDetectionException(MPF_INVALID_PROPERTY, "Could not parse S3_RESULTS_BUCKET '" +
                                                      resultsBucketUrl + "': " + string(ex.what()));
  }

  Aws::Auth::AWSCredentials creds = Aws::Auth::AWSCredentials(Aws::String(accessKey.c_str()),
                                                              Aws::String(secretKey.c_str()));
  s3_client = Aws::S3::S3Client(creds, awsClientConfig,
                                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);

  if (_log->isTraceEnabled()) {
    aws_sdk_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
  } else {
    aws_sdk_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Off;
  }

  Aws::InitAPI(aws_sdk_options);
}

S3StorageHelper::~S3StorageHelper() {
  Aws::ShutdownAPI(aws_sdk_options);
}

bool S3StorageHelper::IsValid() {
  return !s3_bucket.empty();
}

/** ****************************************************************************
* Make a dummy track and reverse transform it so we can get real image coords
* on the fly.  Note a shallow copy of location is made
*
* \param video_cap  mpf video capture object that knows about transforms
* \param frame_idx  frame index of image location to be transformed
* \param loc        image location to be copied and transformed
*
* \returns transformed dummy track containing a transformed copy of loc
***************************************************************************** */
MPFVideoTrack S3StorageHelper::DummyTransform(const MPFVideoCapture &video_cap,
                                              const int frame_idx,
                                              const MPFImageLocation loc) {
  MPFVideoTrack t(frame_idx, frame_idx);
  MPFImageLocation l(loc.x_left_upper, loc.y_left_upper, loc.width, loc.height,
                     loc.confidence, loc.detection_properties);
  t.frame_locations.insert({frame_idx, l});
  video_cap.ReverseTransform(t);
  return t;
}

/** ****************************************************************************
* Grap metadata for s3 object from location
*
* \param    data_uri   source media uri
* \param    model      name oftrtis model used
* \param    loc        location object to pull meta-data from
*
* \returns  collection map to use as s3 meta-data
***************************************************************************** */
// TODO: Refactor this to TrtisDetection class
map<string, string> S3StorageHelper::PrepS3Meta(const string &data_uri,
                                                const string &model,
                                                MPFImageLocation &loc) {
  map<string, string> s3Meta;

  Properties &prop = loc.detection_properties;
  s3Meta.insert({"model", model});
  s3Meta.insert({"data_uri", data_uri});
  s3Meta.insert({"x", to_string(loc.x_left_upper)});
  s3Meta.insert({"y", to_string(loc.y_left_upper)});
  s3Meta.insert({"width", to_string(loc.width)});
  s3Meta.insert({"height", to_string(loc.height)});

  s3Meta.insert({"feature", prop["FEATURE-TYPE"]});
  if (prop.find("CLASSIFICATION") != prop.end()) {
    s3Meta.insert({"class", prop["CLASSIFICATION"]});
    if (loc.confidence > 0) {
      stringstream conf;
      conf << setprecision(2) << fixed << loc.confidence;
      s3Meta.insert({"confidence", conf.str()});
    }
  }

  return s3Meta;
}

/** ****************************************************************************
* Grap metadata for s3 object from video track
*
* \param    data_uri   source media uri
* \param    model      name oftrtis model used
* \param    track      track object to pull meta-data from
* \param    fp_ms      frames per milli-sec for time calculation
*
* \returns  collection map to use as s3 meta-data
***************************************************************************** */
// TODO: Refactor this to TrtisDetection class
map<string, string> S3StorageHelper::PrepS3Meta(const string &data_uri,
                                                const string &model,
                                                MPFVideoTrack &track,
                                                const double &fp_ms) {
  // TODO: This gets the confidence of the first detection only!
  // TODO: Remove need for dummyTransform() by pushing everything to S3 at end of GetDetections(). Doing it before might be wasted time.
  map<string, string> s3Meta = PrepS3Meta(data_uri, model,
                                          track.frame_locations.begin()->second);

  s3Meta.insert({"offsetFrame", to_string(track.start_frame)});
  if (fp_ms > 0.0) {
    stringstream ss;
    ss << setprecision(0) << fixed << track.start_frame / fp_ms;
    s3Meta.insert({"offsetTime", ss.str()});
  }

  return s3Meta;
}

/** ****************************************************************************
* Calculate the sha256 digest for a string buffer
*
* \param    buffer     buffer to calculate the sha256 for
*
* \returns  lowercase hexadecimal string correstponding to the sha256
***************************************************************************** */
string S3StorageHelper::GetSha256(const string &buffer) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, buffer.c_str(), buffer.length());
  SHA256_Final(hash, &sha256);
  std::ostringstream ss;
  ss << hex << setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    ss << setw(2) << static_cast<unsigned>(hash[i]);
  }
  return ss.str();
}

/** ****************************************************************************
* Write a string buffer to an S3 object in an S3 bucket
*
* \param    object_name give the name/key for the object in the bucket
* \param    buffer is the data to write
*
* \returns  URL of object in S3 bucket
***************************************************************************** */
string S3StorageHelper::PutS3Object(const string &buffer,
                                    const std::map<string, string> &metaData) {
  string objectSha = GetSha256(buffer);
  Aws::S3::Model::PutObjectRequest req;
  req.SetBucket(s3_bucket.c_str());
  req.SetKey(objectSha.c_str());
  for (auto &m : metaData) {
    req.AddMetadata(Aws::String(m.first.c_str(), m.first.size()),
                    Aws::String(m.second.c_str(), m.second.size()));
  }
  // see https://stackoverflow.com/questions/48666549/upload-uint8-t-buffer-to-aws-s3-without-going-via-filesystem
  auto data = Aws::MakeShared<Aws::StringStream>("PutObjectInputStream",
                                                 std::stringstream::in
                                                 | std::stringstream::out
                                                 | std::stringstream::binary);
  data->write(reinterpret_cast<char *>(const_cast<char *>(buffer.c_str())), buffer.length());
  req.SetBody(data);
  auto res = s3_client.PutObject(req);

  if (!res.IsSuccess()) {
    stringstream ss;
    ss << res.GetError().GetExceptionName() << ": " << res.GetError().GetMessage();
    throw MPFDetectionException(MPF_FILE_WRITE_ERROR, ss.str());
  }

  return s3_bucket_url + '/' + objectSha;
}

/** ****************************************************************************
* Read a string buffer to an S3 object in an S3 bucket
*
* \param         object_name give the name/key for the object in the bucket
* \param[out]    buffer is where the data will be returned
* \param[in,out] metadata about the object if desired
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::GetS3Object(const string &object_name,
                                  string &buffer) {
  Aws::S3::Model::GetObjectRequest req;
  req.SetBucket(s3_bucket.c_str());
  req.SetKey(object_name.c_str());
  auto res = s3_client.GetObject(req);
  if (res.IsSuccess()) {
    // for alternative to string copy see https://github.com/aws/aws-sdk-cpp/issues/64
    auto &data = res.GetResultWithOwnership().GetBody();
    std::stringstream ss;
    ss << data.rdbuf();
    buffer = ss.str();
    LOG4CXX_TRACE(_log, "Retrieved '" << object_name << "' of size " << buffer.length());
    return true;
  } else { // TODO: Refactor to throw exception.
    auto error = res.GetError();
    LOG4CXX_ERROR(_log, error.GetExceptionName() << ": " << error.GetMessage());
    return false;
  }
}

/******************************************************************************/
bool S3StorageHelper::GetS3Object(const string &object_name,
                                  string &buffer,
                                  map<string, string> &metaData) {
  Aws::S3::Model::GetObjectRequest req;
  req.SetBucket(s3_bucket.c_str());
  req.SetKey(object_name.c_str());
  auto res = s3_client.GetObject(req);
  if (res.IsSuccess()) {
    // for alternative to string copy see https://github.com/aws/aws-sdk-cpp/issues/64
    auto &data = res.GetResultWithOwnership().GetBody();
    std::stringstream ss;
    ss << data.rdbuf();
    buffer = ss.str();
    LOG4CXX_TRACE(_log, "Retrieved '" << object_name << "' of size " << buffer.length());
    for (auto &p : res.GetResult().GetMetadata()) {
      metaData[string(p.first.c_str(), p.first.size())] = string(p.second.c_str(), p.second.size());
    }
    return true;
  } else { // TODO: Refactor to throw exception.
    auto error = res.GetError();
    LOG4CXX_ERROR(_log, error.GetExceptionName() << ": " << error.GetMessage());
    return false;
  }
}

/** ****************************************************************************
* Delete an object out of an S3 bucket
*
* \param    object_name give the name/key for the object in the bucket
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::DeleteS3Object(const string &object_name) {
  Aws::S3::Model::DeleteObjectRequest req;
  req.SetBucket(s3_bucket.c_str());
  req.SetKey(object_name.c_str());
  auto res = s3_client.DeleteObject(req);
  if (!res.IsSuccess()) { // TODO: Refactor to throw exception.
    auto error = res.GetError();
    LOG4CXX_ERROR(_log, "Could not delete object '" << object_name << "':"
                                                    << error.GetExceptionName() << ": " << error.GetMessage());
    return false;
  }
  return true;
}

/** ****************************************************************************
* Check if an object exists in an S3 bucket
*
* \param    object_name give the name/key for the object in the bucket
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::ExistsS3Object(const string &object_name) {
  Aws::S3::Model::HeadObjectRequest req;
  req.SetBucket(s3_bucket.c_str());
  req.SetKey(object_name.c_str());
  const auto res = s3_client.HeadObject(req);
  return res.IsSuccess();
}

/** ****************************************************************************
* Check if a bucket exists in an S3 store
*
* \param    bucket_name give the name of the bucket into which to write
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::ExistsS3Bucket(const string &bucket_name) {
  Aws::S3::Model::HeadBucketRequest req;
  const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
  req.SetBucket(bucket.c_str());
  const auto res = s3_client.HeadBucket(req);
  return res.IsSuccess();
}

/** ****************************************************************************
* Create a bucket in an S3 store if it does not exist
*
* \param    bucket_name give the name of the bucket into which to write
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::CreateS3Bucket(const string &bucket_name) {
  const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
  if (ExistsS3Bucket(bucket)) {
    LOG4CXX_TRACE(_log, "Bucket '" << bucket << "' already exists");
    return true;
  }

  Aws::S3::Model::CreateBucketRequest req;
  req.SetBucket(bucket.c_str());
  const auto res = s3_client.CreateBucket(req);
  if (!res.IsSuccess()) { // TODO: Refactor to throw exception.
    auto error = res.GetError();
    LOG4CXX_ERROR(_log, "Unable to create bucket '" << bucket << "': "
                                                    << error.GetExceptionName() << ": " << error.GetMessage());
    return false;
  }

  return true;
}

/** ****************************************************************************
* Delete a bucket in an S3 store if it exists
*
* \param    bucket_name give the name of the bucket into which to write
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::DeleteS3Bucket(const string &bucket_name) {
  const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
  if (!ExistsS3Bucket(bucket)) {
    LOG4CXX_TRACE(_log, "Bucket '" << bucket << "' does not exist");
    return true;
  }

  Aws::S3::Model::DeleteBucketRequest req;
  req.SetBucket(bucket.c_str());
  const auto res = s3_client.DeleteBucket(req);
  if (!res.IsSuccess()) { // TODO: Refactor to throw exception.
    auto error = res.GetError();
    LOG4CXX_ERROR(_log, "Unable to delete bucket '" << bucket << "': "
                                                    << error.GetExceptionName() << ": " << error.GetMessage());
    return false;
  }

  return true;
}

/** ****************************************************************************
* Empty a bucket in an S3 store if it exists
*
* \param    bucket_name give the name of the bucket into which to write
*
* \returns  true if successful
***************************************************************************** */
bool S3StorageHelper::EmptyS3Bucket(const string &bucket_name) {
  const string bucket = bucket_name.empty() ? s3_bucket : bucket_name;
  if (!ExistsS3Bucket(bucket)) {
    LOG4CXX_TRACE(_log, "Bucket '" << bucket << "' does not exist");
    return true;
  }

  Aws::S3::Model::ListObjectsV2Request req;
  req.SetBucket(bucket.c_str());
  while (true) {
    auto res = s3_client.ListObjectsV2(req);
    Aws::Vector<Aws::S3::Model::Object> oList = res.GetResult().GetContents();
    if (oList.size() > 0) {
      Aws::S3::Model::Delete dObjs;
      //see https://github.com/aws/aws-sdk-cpp/issues/91
      for (auto const &o: oList) {
        Aws::S3::Model::ObjectIdentifier oId;
        oId.SetKey(o.GetKey());
        dObjs.AddObjects(oId);
      }
      Aws::S3::Model::DeleteObjectsRequest dReq;
      dReq.SetBucket(bucket.c_str());
      dReq.SetDelete(dObjs);
      auto dRes = s3_client.DeleteObjects(dReq);
      if (!dRes.IsSuccess()) { // TODO: Refactor to throw exception.
        auto error = res.GetError();
        LOG4CXX_TRACE(_log, "Could delete all files in bucket '" << bucket << "': "
                                                                 << error.GetExceptionName() << ": "
                                                                 << error.GetMessage());
        return false;
      }
    } else {
      break;
    }
  }
  // extra stuff needs to happen if versioned see https://docs.aws.amazon.com/AmazonS3/latest/dev/delete-or-empty-bucket.html#empty-bucket-awssdks
#ifdef VERSIONED_S3_OBJECTS
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
  return true;
}
