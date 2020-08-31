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

#include <string>

#include <log4cxx/consoleappender.h>
#include <log4cxx/simplelayout.h>

#include <gtest/gtest.h>

#include "S3StorageUtil.h"

using namespace MPF::COMPONENT;
using namespace std;

S3StorageUtil _init() {
    auto log = log4cxx::Logger::getLogger("S3StorageUtil");
    auto appender = new log4cxx::ConsoleAppender(new log4cxx::SimpleLayout());
    log->addAppender(appender);
    log->setLevel(log4cxx::Level::getTrace());

    string resultsBucketUrl = "http://minio:9000/test-bucket";
    return {log, resultsBucketUrl, "minio", "minio123"};
}

TEST(S3Storage, TestDefaultBucket) {
    S3StorageUtil &&s3StorageUtil = _init();

    ASSERT_EQ("http://minio:9000/test-bucket", s3StorageUtil.GetS3ResultsBucketUrl());
    ASSERT_EQ("http://minio:9000", s3StorageUtil.GetS3ResultsEndpoint());
    ASSERT_EQ("test-bucket", s3StorageUtil.GetS3ResultsBucket());

    // The following operations should not throw if the bucket doesn't exist:
    s3StorageUtil.EmptyS3Bucket();
    s3StorageUtil.DeleteS3Bucket();

    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket());
    s3StorageUtil.CreateS3Bucket();
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket());
    s3StorageUtil.CreateS3Bucket(); // should not throw

    string val = "foo";
    string sha = s3StorageUtil.GetSha256(val);
    map<string, string> metaIn({{"meta-foo-key", "meta-foo-val"}});

    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(sha));
    s3StorageUtil.PutS3Object(val, metaIn);
    ASSERT_TRUE(s3StorageUtil.ExistsS3Object(sha));

    string buffer;
    map<string, string> metaOut;
    s3StorageUtil.GetS3Object(sha, buffer, metaOut);
    ASSERT_EQ(val, buffer);
    ASSERT_EQ(metaIn, metaOut);

    s3StorageUtil.DeleteS3Object(sha);
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(sha));

    metaIn.clear();
    metaOut.clear();

    val = "bar";
    sha = s3StorageUtil.GetSha256(val);
    metaIn.insert({"meta-bar-key", "meta-bar-val"});
    string url = s3StorageUtil.PutS3Object(val, metaIn);
    s3StorageUtil.GetS3Object(sha, buffer, metaOut);
    ASSERT_EQ(val, buffer);
    ASSERT_EQ(metaIn, metaOut);
    ASSERT_EQ("http://minio:9000/test-bucket/" + sha, url);

    val = "bar-nometa";
    sha = s3StorageUtil.GetSha256(val);
    s3StorageUtil.PutS3Object(val);
    s3StorageUtil.GetS3Object(sha, buffer);
    ASSERT_EQ(val, buffer);

    s3StorageUtil.EmptyS3Bucket();
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(sha));
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket());

    s3StorageUtil.DeleteS3Bucket();
    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket());
}

TEST(S3Storage, TestOtherBucket) {
    S3StorageUtil &&s3StorageUtil = _init();

    string s3Bucket = "animal-bucket";

    // The following operations should not throw if the bucket doesn't exist:
    s3StorageUtil.EmptyS3Bucket(s3Bucket);
    s3StorageUtil.DeleteS3Bucket(s3Bucket);

    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket(s3Bucket));
    s3StorageUtil.CreateS3Bucket(s3Bucket);
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket(s3Bucket));
    s3StorageUtil.CreateS3Bucket(s3Bucket); // should not throw

    string val = "dog";
    string sha = s3StorageUtil.GetSha256(val);
    map<string, string> metaIn({{"meta-dog-key", "meta-dog-val"}});

    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(s3Bucket, sha));
    s3StorageUtil.PutS3Object(s3Bucket, val, metaIn);
    ASSERT_TRUE(s3StorageUtil.ExistsS3Object(s3Bucket, sha));
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(sha)); // not in default bucket

    string buffer;
    map<string, string> metaOut;
    s3StorageUtil.GetS3Object(s3Bucket, sha, buffer, metaOut);
    ASSERT_EQ(val, buffer);
    ASSERT_EQ(metaIn, metaOut);

    s3StorageUtil.DeleteS3Object(s3Bucket, sha);
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(s3Bucket, sha));

    metaIn.clear();
    metaOut.clear();

    val = "cat";
    sha = s3StorageUtil.GetSha256(val);
    metaIn.insert({"meta-cat-key", "meta-cat-val"});
    string url = s3StorageUtil.PutS3Object(s3Bucket, val, metaIn);
    s3StorageUtil.GetS3Object(s3Bucket, sha, buffer, metaOut);
    ASSERT_EQ(val, buffer);
    ASSERT_EQ(metaIn, metaOut);
    ASSERT_EQ("http://minio:9000/animal-bucket/" + sha, url);

    val = "cat-nometa";
    sha = s3StorageUtil.GetSha256(val);
    s3StorageUtil.PutS3Object(s3Bucket, val);
    s3StorageUtil.GetS3Object(s3Bucket, sha, buffer);
    ASSERT_EQ(val, buffer);

    s3StorageUtil.EmptyS3Bucket(s3Bucket);
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(s3Bucket, sha));
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket(s3Bucket));

    s3StorageUtil.DeleteS3Bucket(s3Bucket);
    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket(s3Bucket));
}