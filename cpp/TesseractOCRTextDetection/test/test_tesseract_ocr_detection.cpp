/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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
#include <MPFDetectionComponent.h>

#include <gtest/gtest.h>

#include "TesseractOCRTextDetection.h"

using namespace MPF::COMPONENT;


MPFImageJob createOCRJob(const std::string &uri){
	Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    algorithm_properties["TAGGING_FILE"] = "text-tags.json";
    algorithm_properties["SHARPEN"] = "1.0";
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}


bool containsObject(const std::string &object_name, const std::vector<MPFImageLocation> &locations) {
    for (int i = 0; i < locations.size(); i++) {

        std::string text = locations[i].detection_properties.at("TEXT");
        if(text.find(object_name)!= std::string::npos)
            return true;
    }
    return false;
}

void assertObjectDetectedInImage(const std::string &expected_object, const std::string &image_path,
                                 TesseractOCRTextDetection &ocr) {
    MPFImageJob job = createOCRJob(image_path);

    std::vector<MPFImageLocation> image_locations;
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_object, image_locations))
                                << "Expected OCR to detect text \"" << expected_object << "\" in " << image_path;
}
void assertObjectNotDetectedInImage(const std::string &expected_object, const std::string &image_path,
                                 TesseractOCRTextDetection &ocr) {
    MPFImageJob job = createOCRJob(image_path);

    std::vector<MPFImageLocation> image_locations;
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());

    ASSERT_FALSE(containsObject(expected_object, image_locations))
                                << "Expected OCR to NOT detect text \"" << expected_object << "\" in " << image_path;
}

TEST(TESSERACTOCR, ImageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocr.Init());

    assertObjectDetectedInImage("TESTING 123", "test/text-demo.png", ocr);
    assertObjectNotDetectedInImage("Ponies", "test/text-demo.png", ocr);

    ASSERT_TRUE(ocr.Close());
}







