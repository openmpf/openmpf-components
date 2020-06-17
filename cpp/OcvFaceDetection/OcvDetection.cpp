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

/* GroupRectanglesMod is a modified version of groupRectangles included in the
   OpenCV source and for that reason the copyright notice below is provided */

////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////////////


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "OcvDetection.h"


using std::pair;
using std::vector;

using cv::Mat;
using cv::Rect;
using cv::saturate_cast;
using cv::SimilarRects;
using cv::Size;

OcvDetection::OcvDetection() {
    initialized = false;

    //trained lbp cascade
    face_cascade_path = "/data/cascade.xml";
}

OcvDetection::~OcvDetection() { }

bool OcvDetection::Init(std::string &plugin_path) {
    openFaceDetectionLogger = log4cxx::Logger::getLogger("OcvFaceDetection");

    //Load cascades
    std::string cascade_path = plugin_path + face_cascade_path;
    if( !face_cascade.load(cascade_path) ) {
        LOG4CXX_ERROR(openFaceDetectionLogger, "Issue loading: " << cascade_path);
        return false;
    }

    initialized = true;

    return true;
}

/*This method clusters all the input rectangles (rectList) using the rectangle equivalence criteria that
combines rectangles with similar sizes and similar locations. */
void OcvDetection::GroupRectanglesMod(vector<Rect>& rectList, int groupThreshold, double eps,
                                      vector<int>* weights, vector<double>* levelWeights) {
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }

    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
                         saturate_cast<int>(r.y*s),
                         saturate_cast<int>(r.width*s),
                         saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        //the change from source
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

vector<pair<Rect, int>> OcvDetection::DetectFaces(const Mat &frame_gray, int min_face_size) {
    std::vector<Rect> faces;
    std::vector<Rect> weighted_faces;
    std::vector<pair<Rect, int>> face_confidence_pairs;

    if(!initialized) {
        LOG4CXX_ERROR(openFaceDetectionLogger, "Ocv detection is not initialized and cannot detect faces. Please check previous errors.");
        return face_confidence_pairs;
    }

    Mat frame_gray_clone = frame_gray.clone();
    equalizeHist( frame_gray_clone, frame_gray_clone);

    vector<int> reject_levels;
    vector<double> level_weights;

    //these will be used if detecting with and without weights and combining results
    float scale_factor = 1.4f;
    int min_neighbors = 6;
    int flags = 0;
    double GROUP_EPS = 0.95;

    //commenting out the code supporting this because it may be useful, but is not currently needed
//	bool multi_detect_join = false;

//	if(multi_detect_join) {
//		face_cascade.detectMultiScale(frame_gray_clone, weighted_faces, reject_levels, level_weights, scale_factor, min_neighbors, 0, Size(min_face_size, min_face_size), Size(), true); //, true);
//	} else {
    scale_factor = 1.2f;
    //min_neighbors is set to -1 to trick ocv source along with disabling reject levels and modified rectangle grouping (GroupRectanglesMod)
    //groupRectangles gets called at the end of detectMultiScale if neighbors is positive and that's
    //why a -1 is passed - this allows the use of the version of groupRectangles in this class (GroupRectanglesMod)
    min_neighbors = -1;
    GROUP_EPS = 0.2;
    face_cascade.detectMultiScale(frame_gray_clone,
                                  weighted_faces,
                                  scale_factor,
                                  min_neighbors,
                                  0,
                                  Size(min_face_size, min_face_size),
                                  Size());

    //back to a more normal value for the grouping
    min_neighbors = 4;
    GroupRectanglesMod( weighted_faces, min_neighbors, GROUP_EPS, &reject_levels, &level_weights);
//	}

    if(weighted_faces.size() > 0) {

        //this is calling the old (built form source) group rectangles method
//		if(multi_detect_join) {
//			face_cascade.detectMultiScale(frame_gray_clone, faces, 1.1, 4, 0, Size(min_face_size, min_face_size));
//			groupRectangles( weighted_faces, reject_levels, level_weights, min_neighbors, GROUP_EPS );
//		}

        for (int i = 0; i < weighted_faces.size(); i++) {
            Rect existing_rect = weighted_faces[i];

            Rect found_rect(0, 0, 0, 0);

//			if(multi_detect_join) {
//				for (int j = 0; j < faces.size(); j++) {
//
//					Rect new_rect = faces[j];
//
//					Rect intersection = existing_rect & new_rect; // (rectangle intersection)
//
//					//should loop through all intersecting rects and choose the one with the most intersection
//					//and that will be the correct intersection
//					//important: should allow for a small intersection for faces in close proximity - like 10 - 20%
//					if (intersection.area() > ceil(static_cast<float>(existing_rect.area()) * 0.15)) {
//						found_rect = new_rect;
//						break;
//					}
//				}
//			} else {
            found_rect = existing_rect;
//			}

            if(found_rect.area() > 0) {
                int reject_level = reject_levels[i];
                pair<Rect, int> p(found_rect, reject_level);
                face_confidence_pairs.push_back(p);
            }
        }
    }

    return face_confidence_pairs;
}
