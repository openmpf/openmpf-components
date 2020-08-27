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

TEST(TRTIS, S3Test) {
    auto log = log4cxx::Logger::getLogger("S3StorageUtil");
    auto appender = new log4cxx::ConsoleAppender(new log4cxx::SimpleLayout());
    log->addAppender(appender);
    log->setLevel(log4cxx::Level::getTrace());

    string resultsBucketUrl = "http://minio:9000/trtis-test";
    S3StorageUtil s3StorageUtil(log, resultsBucketUrl, "minio", "minio123");

    ASSERT_EQ(s3StorageUtil.GetS3ResultsBucket(),"trtis-test");

    // The following operations should not throw if the bucket doesn't exist:
    s3StorageUtil.EmptyS3Bucket();
    s3StorageUtil.DeleteS3Bucket();

    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket());
    s3StorageUtil.CreateS3Bucket();
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket());
    s3StorageUtil.CreateS3Bucket(); // should not throw when the bucket doesn't exist

    string val = "foo";
    string sha = s3StorageUtil.GetSha256(val);
    map<string, string> metaIn({{"meta1", "meta2"}});

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
    metaIn.insert({"meta3", "meta4"});
    string url = s3StorageUtil.PutS3Object(val, metaIn);
    s3StorageUtil.GetS3Object(sha, buffer, metaOut);
    ASSERT_EQ(val, buffer);
    ASSERT_EQ(metaIn, metaOut);
    ASSERT_EQ(url, resultsBucketUrl + "/" + sha);

    val = "nometa";
    sha = s3StorageUtil.GetSha256(val);
    s3StorageUtil.PutS3Object(val);
    s3StorageUtil.GetS3Object(sha, buffer);
    ASSERT_EQ(val, buffer);

    s3StorageUtil.EmptyS3Bucket();
    ASSERT_FALSE(s3StorageUtil.ExistsS3Object(sha));
    ASSERT_TRUE(s3StorageUtil.ExistsS3Bucket());

    s3StorageUtil.DeleteS3Bucket();
    ASSERT_FALSE(s3StorageUtil.ExistsS3Bucket());

    // TODO: Test with no provided S3 bucket on creation / or remove functionality.

    /*
    s3StorageUtil.EmptyS3Bucket();
    ASSERT_TRUE(trtisDet._awsDeleteS3Bucket(cfg,bucket));
    ASSERT_FALSE(trtisDet._awsExistsS3Bucket(cfg));
    ASSERT_TRUE(trtisDet._awsCreateS3Bucket(cfg));
    ASSERT_TRUE(trtisDet._awsExistsS3Bucket(cfg));
    ASSERT_TRUE(trtisDet._awsCreateS3Bucket(cfg));

    map<string,string> metaData;
    metaData["mfoo"] = "mbar";
    ASSERT_TRUE(trtisDet._awsPutS3Object(cfg,"foo-bar","hello",metaData));

    ASSERT_TRUE(trtisDet._awsExistsS3Object(cfg,"foo-bar"));
    ASSERT_TRUE(trtisDet._awsDeleteS3Object(cfg,"foo-bar"));
    ASSERT_FALSE(trtisDet._awsExistsS3Object(cfg,"foo-bar"));
    ASSERT_TRUE(trtisDet._awsDeleteS3Bucket(cfg));
    ASSERT_FALSE(trtisDet._awsExistsS3Bucket(cfg));
    ASSERT_TRUE(trtisDet._awsCreateS3Bucket(cfg));
    ASSERT_TRUE(trtisDet._awsPutS3Object(cfg,"foo-bar","hello",metaData));

    string buffer;
    map<string,string> metaData2;
    ASSERT_TRUE(trtisDet._awsGetS3Object(cfg,"foo-bar",buffer,metaData2));
    ASSERT_EQ(buffer,"hello");
    ASSERT_EQ(metaData2,metaData);
    ASSERT_TRUE(trtisDet.Close());
    */
}