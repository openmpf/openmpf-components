/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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

#include "OcvFaceDetection.h"

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <QFile>
#include <QFileInfo>

#include "Utils.h"
#include "detectionComponentUtils.h"
#include "MPFSimpleConfigLoader.h"
#include "MPFImageReader.h"


using std::string;
using std::vector;
using std::map;
using std::pair;
using std::runtime_error;

using cv::Point;
using cv::Point2f;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
using cv::Size;
using cv::RotatedRect;
using cv::Size2f;
using cv::namedWindow;
using cv::destroyAllWindows;
using cv::waitKey;
using cv::KeyPoint;
using cv::resize;
using cv::rectangle;

using log4cxx::Logger;

using namespace MPF;
using namespace COMPONENT;



string OcvFaceDetection::GetDetectionType() {
    return "FACE";
}

void OcvFaceDetection::SetModes(bool display_window, bool print_debug_info) {
    imshow_on = display_window;

    if (print_debug_info && OpenFaceDetectionLogger != NULL) {
        OpenFaceDetectionLogger->setLevel(log4cxx::Level::getDebug());
    }
}

bool OcvFaceDetection::Init() {
    string run_dir = GetRunDirectory();
    string plugin_path = run_dir + "/OcvFaceDetection";
    string config_path = plugin_path + "/config";

    OpenFaceDetectionLogger = log4cxx::Logger::getLogger("OcvFaceDetection");

    //must initialize opencv face detection
    if (!ocv_detection.Init(plugin_path)) {
        LOG4CXX_ERROR(OpenFaceDetectionLogger, "Failed to initialize OpenCV Detection");
        return false;
    }

    SetDefaultParameters();

    //once this is done - parameters will be set and SetReadConfigParameters() can be called again to revert back
    //to the params read at intialization
    string config_params_path = config_path + "/mpfOcvFaceDetection.ini";
    int rc = LoadConfig(config_params_path, parameters);
    if (rc) {
        LOG4CXX_ERROR(OpenFaceDetectionLogger, "Failed to load the OcvFaceDetection config from: " << config_params_path);
        return (false);
    }

    SetReadConfigParameters();

    return true;
}

bool OcvFaceDetection::Close() {
    CloseWindows();
    return true;
}

void OcvFaceDetection::SetDefaultParameters() {
    max_features = 250;

    //limiting the number of corners to 250 by default
    feature_detector = cv::GFTTDetector::create(max_features);

    min_face_size = 48;

    //this should be adjusted based on type of detector
    min_init_point_count = 45;

    //point at which the track should be considered lost
    min_point_percent = 0.70;

    //the point at which points will be redetected
    min_redetect_point_perecent = 0.88;

    min_initial_confidence = 10.0f;

    //not currently used - but could be used to help stops tracks earlier when there is a lot
    //of error when detecting the next points using calcopticalflow
    max_optical_flow_error = 4.7;;
}

void OcvFaceDetection::SetReadConfigParameters() {
    //make sure none of the parameters are missed in the config file - double check
    imshow_on = parameters["IMSHOW_ON"].toInt();

    min_init_point_count = parameters["MIN_INIT_POINT_COUNT"].toInt();

    min_point_percent = parameters["MIN_POINT_PERCENT"].toFloat();
    min_initial_confidence = parameters["MIN_INITIAL_CONFIDENCE"].toFloat();

    min_face_size = parameters["MIN_FACE_SIZE"].toInt();

    //right now only accepting a verbosity of 1 and just checking for > 0, may need to adjust later
    //if verbosity 1 set the log level to DEBUG
    //if verbosity set to 2, think about using TRACE
    verbosity = parameters["VERBOSE"].toInt();
    if (verbosity > 0) {
        OpenFaceDetectionLogger->setLevel(log4cxx::Level::getDebug());
    }
}

/* This function reads a property value map and adjusts the settings for this component. */
void OcvFaceDetection::GetPropertySettings(const map <string, string> &algorithm_properties) {
    string property;
    string str_value;
    int ivalue;
    float fvalue;
    for (map<string, string>::const_iterator imap = algorithm_properties.begin();
         imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;

        //TODO: could restrict some of the parameter ranges here and log like PP!
        if (property == "MIN_FACE_SIZE") { //INT
            min_face_size = atoi(str_value.c_str());
        }
        else if (property == "MAX_FEATURE") { //INT
            max_features = atoi(str_value.c_str());
        }
        if (property == "MIN_INIT_POINT_COUNT") { //INT
            min_init_point_count = atoi(str_value.c_str());
        }
        else if (property == "MIN_POINT_PERCENT") { //FLOAT
            min_point_percent = atof(str_value.c_str());
        }
        else if (property == "MIN_INITIAL_CONFIDENCE") { //FLOAT
            min_initial_confidence = atof(str_value.c_str());
        }
        else if (property == "MAX_OPTICAL_FLOW_ERROR") { //FLOAT
            max_optical_flow_error = atof(str_value.c_str());
        }
        else if (property == "VERBOSE") { //INT
            verbosity = atoi(str_value.c_str());
        }
    }
    return;
}

void OcvFaceDetection::Display(const string title, const Mat &img) {
    if (imshow_on) {
        imshow(title, img);
        waitKey(5);
    }
}

Rect OcvFaceDetection::GetMatch(const Mat &frame_rgb_display, const Mat &frame_gray, const Mat &templ) {
    //no clue what method is best - default of the opencv demo
    int match_method = cv::TM_CCOEFF_NORMED;

    Mat img_display = frame_rgb_display.clone();

    /// Create the result matrix - equals the size of the search window using the
    //left corner of the template mat
    int result_cols = frame_gray.cols - templ.cols + 1;
    int result_rows = frame_gray.rows - templ.rows + 1;

    Mat result(result_cols, result_rows, CV_32FC1);

    /// Do the Matching and Normalize
    matchTemplate(frame_gray, templ, result, match_method);
    normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, Mat());

    /// Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED) { matchLoc = minLoc; }
    else { matchLoc = maxLoc; }

    Rect match_rect(matchLoc.x, matchLoc.y, templ.cols, templ.rows);

    ///show the location of the detected template with the highest match
    //rectangle(img_display, match_rect, Scalar::all(0), 2, 8, 0 );
    //imshow( "img_display", img_display );

    return match_rect;
}

bool OcvFaceDetection::IsExistingTrackIntersection(const Rect new_rect, int &intersection_index) {
    intersection_index = -1;

    for (vector<Track>::iterator track = current_tracks.begin(); track != current_tracks.end(); ++track) {
        ++intersection_index;

        //if the track is new then there should still be a face detection
        MPFImageLocation last_face_detection = MPFImageLocation(track->face_track.frame_locations.rbegin()->second); // last element in map

        Rect existing_rect = Utils::ImageLocationToCvRect(last_face_detection);

        Rect intersection = existing_rect & new_rect; // (rectangle intersection)

        //important: just returning the index with any sort of intersection will cause issues
        //if there are faces in close proximity and there is more than one intersection
        //need to loop through all intersecting rects and choose the one with the most intersection
        //and that will be the correct intersection index
        //important: should allow for a small intersection for faces in close proximity - like 10 - 20%
        if (intersection.area() > ceil(static_cast<float>(existing_rect.area()) * 0.15)) {
            return true;
        }
    }
    return false;
}

Rect OcvFaceDetection::GetUpscaledFaceRect(const Rect &face_rect) {
    return Rect(
            face_rect.x + static_cast<int>( -0.214 * static_cast<float>(face_rect.width)),
            face_rect.y + static_cast<int>( -0.055 * static_cast<float>(face_rect.height)),
            static_cast<int>( 1.4286 * static_cast<float>(face_rect.width)),
            static_cast<int>( 1.11 * static_cast<float>(face_rect.height)));
}

Mat OcvFaceDetection::GetMask(const Mat &frame, const Rect &face_rect, bool copy_face_rect) {
    //create a single channel zero matrix the size of the frame
    Mat image_mask;
    image_mask = Mat::zeros(frame.size(), CV_8UC1);

    //Downsize the bounding box so that only the 'face' is shown
    Rect rescaled_face = Rect(
            face_rect.x + static_cast<int>( 0.15 * static_cast<float>(face_rect.width)),
            face_rect.y + static_cast<int>( 0.05 * static_cast<float>(face_rect.height)),
            static_cast<int>( 0.7 * static_cast<float>(face_rect.width)),
            static_cast<int>( 0.9 * static_cast<float>(face_rect.height)));

    //crop the frame to the face, resize and display
    Mat face_roi = Mat(frame, rescaled_face);
    Mat face_roi_resize;
    resize(face_roi, face_roi_resize, Size(256, 256));

    //using a best fit ellipse in an attempt to remove any "non-face" image parts from the face bounding box
    //find the center of the rescaled face
    Point2f center(static_cast<float>( rescaled_face.x + 0.5 * static_cast<float>(rescaled_face.width)),
                   static_cast<float>( rescaled_face.y + 0.5 * static_cast<float>(rescaled_face.height)));
    RotatedRect rotated_rect;
    rotated_rect.center = center;
    rotated_rect.size = Size2f(static_cast<float>(rescaled_face.width), static_cast<float>(rescaled_face.height));
    //finding the face angle would greatly improve the results of the initial point detection
    //would allow for using a rotated rect!! - possible future improvement
    rotated_rect.angle = 0.0f;
    //draw the ellipse on the black frame to create the mask
    ellipse(image_mask, rotated_rect, Scalar(255), cv::FILLED);

    if (copy_face_rect) {
        //Copy face to masked image
        frame.copyTo(image_mask, image_mask);
    }

    return image_mask;
}

bool OcvFaceDetection::IsBadFaceRatio(const Rect &face_rect) {
    //trying to find a way to kill tracks when points grab onto something outside of the face and the face
    //bounding box ratio becomes odd
    //if bounding rect width is much more than the height there is an issue or if the area of the enclosing circle
    //is much greater than the bounding rect area

    float face_ratio = static_cast<float>(face_rect.width) / static_cast<float>(face_rect.height);
    float target_face_ratio = 0.75;
    //the threshold for greater than should be bigger than the threshold for less than - face rects could be very thin if equal
    float max_increase_face_ratio_deviation = 0.35;
    float max_decrease_face_ratio_deviation = -0.25;
    float face_ratio_diff = face_ratio - target_face_ratio;
    //should check this ratio at track creation..

    if ((face_ratio_diff > max_increase_face_ratio_deviation ||
         face_ratio_diff < max_decrease_face_ratio_deviation)) {
        return true;
    }

    return false;
}

void OcvFaceDetection::CloseAnyOpenTracks(int frame_index) {
    if (!current_tracks.empty()) {
        //need to stop all current tracks!
        for (vector<Track>::iterator it = current_tracks.begin(); it != current_tracks.end(); it++) {
            //grab last track
            Track track(*it);
            //should never happen - but ignoring the track
            if (track.face_track.stop_frame != -1) {
                continue;
            }

            //track is still going at end index
            //set the stopFrame for this track!!!
            track.face_track.stop_frame = frame_index;

            //now the track can be saved
            saved_tracks.push_back(track);
        }
    }
}

void OcvFaceDetection::AdjustRectToEdges(Rect &rect, const Mat &src) {
    if (!src.empty()) {
        //check corners and edges and resize appropriately!

        //modifying the referenced rect
        //subtracting 1 since indexes are 0 based, if image is 256x256 there is will be 256 rows and cols,
        //but the values will range from 0 to 255
        int x_max = (src.cols - 1);
        int y_max = (src.rows - 1);

        //x
        int x_adj = 0;
        if (rect.x < 0) {
            //subtract the negative value
            x_adj = 0 - rect.x;
            rect.x = 0;
        }
            //rect.x should never be greater than image bounds, the bounding box would start outside of the image
        else if ((rect.x + rect.width) > x_max) {
            x_adj = (rect.x + rect.width) - x_max;;
        }

        //y
        int y_adj = 0;
        if (rect.y < 0) {
            //subtract the negative value
            y_adj = 0 - rect.y;
            rect.y = 0;
        }
            //rect.y should never be greater than image bounds, the bounding box would start outside of the image
        else if ((rect.y + rect.height) > y_max) {
            y_adj = (rect.y + rect.height) - y_max;
        }

        //now adjust width and height
        //adjustment values shouldn't be negative
        if (x_adj > 0) {
            rect.width = rect.width - x_adj;
        }

        if (y_adj > 0) {
            rect.height = rect.height - y_adj;
        }

        //the rect may still have a width and/or height greater than src and that must be adjusted
        if (rect.height > src.rows) {
            rect.height = src.rows;
        }

        if (rect.width > src.cols) {
            rect.width = src.cols;
        }
    }
}

vector<MPFVideoTrack> OcvFaceDetection::GetDetections(const MPFVideoJob &job) {
    try {
        SetDefaultParameters();
        SetReadConfigParameters();
        GetPropertySettings(job.job_properties);

        MPFVideoCapture video_capture(job, true, true);

        vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(job, video_capture);

        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }
        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, OpenFaceDetectionLogger);
    }
}



vector<MPFVideoTrack> OcvFaceDetection::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture) {


    long total_frames = video_capture.GetFrameCount();
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Total video frames: " << total_frames);

    int frame_index = 0;

    if (imshow_on) {
        namedWindow("Open Tracker", 0);
    }

    Mat frame, frame_draw, frame_draw_pre_verified;
    //need to store the previous frame
    Mat gray, prev_gray;

    while (video_capture.Read(frame)) {

        if (imshow_on) {
            //create copy of frame to draw on for display
            frame_draw = frame.clone();
        }

        //Convert to grayscale
        gray = Utils::ConvertToGray(frame);

        //look for new faces
        vector <pair<Rect, int>> faces = ocv_detection.DetectFaces(gray, min_face_size);
        //draw all new face bounding boxes on the screen for debugging
        if (imshow_on) {
            for (vector <pair<Rect, int>>::iterator face_rect = faces.begin(); face_rect != faces.end(); ++face_rect) {
                rectangle(frame_draw, ((*face_rect).first), Scalar(204, 0, 204), 2, 4);
            }
        }

        int track_index = -1;
        for (vector<Track>::iterator track = current_tracks.begin(); track != current_tracks.end(); ++track) {
            ++track_index;
            //assume track lost at start
            track->track_lost = true;

            if (track->previous_points.empty()) {
                //should never get here!! - current points of first detection will be swapped to previous!!
                //kill the track if this case does occur
                LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                           << "] Track contains no previous points - killing tracks");
                continue;
            }

            vector <Point2f> new_points;
            vector <uchar> status;
            vector <float> err;
            //get new points
            calcOpticalFlowPyrLK(prev_gray, gray, track->previous_points,
                                 new_points, status, err); //, win_size, 3, term_criteria, 0, 0.001);

            //stores the detected rect properly matched up to the current track
            //if certain requirements are met
            pair<Rect, int> correct_detected_rect_pair;
            correct_detected_rect_pair.first = Rect(0, 0, 0, 0);

            //set to true if template matching is used to keep the track going
            //this will be used to determine if points can be redetected
            bool track_recovered = false;

            if (new_points.empty()) {
                //no new points - this track will be killed
                LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                           << "] Optical flow could not find any new points - killing track");
                continue;
            }
            else {
                //don't want to display any of the drawn points or bounding boxes of tracks that aren't kept
                frame_draw_pre_verified = frame_draw.clone();

                //store the correct bounding box here
                //Rect final_bounding_box(0,0,0,0);

                //GOAL: try to find the current detection with the most contained points
                //the goal is to use the opencv bounding box rather than estimate one using
                //an enclosed circle and a bounding rect
                //it will also be good to remove points not in this bounding box to improve the continued tracking
                //Rect correct_detected_rect(0,0,0,0);

                int points_within_detected_rect = 0;
                //store the percentage of intersection for each detected face
                map <int, float> rect_point_percentage_map;
                int face_rect_index = 0;
                for (vector <pair<Rect, int>>::iterator face_rect = faces.begin(); face_rect != faces.end(); ++face_rect) {
                    points_within_detected_rect = 0;

                    for (unsigned i = 0; i < new_points.size(); i++) { //TODO: could also use the err vector (from calcOpticalFlowPyrLK) with a float threshold
                        if (face_rect->first.contains(new_points[i])) {
                            ++points_within_detected_rect;
                        }
                    }

                    //if points were found inside one of the rects then it needs to be stored in the map
                    if (points_within_detected_rect > 0) {
                        rect_point_percentage_map[face_rect_index] = (static_cast<float>(points_within_detected_rect) /
                                                                      static_cast<float>(new_points.size()));
                    }

                    ++face_rect_index;
                }

                //TODO: could combine this with the loop above
                //also probably want the best intersection percentage to be > 75 or 80 percent
                // - if only a few points are intersecting then there is clearly an issue
                //float best_intersection_percentage = 0.0;
                //must be greater than 0.75
                float best_intersection_percentage = 0.75;
                for (map<int, float>::iterator map_iter = rect_point_percentage_map.begin();
                     map_iter != rect_point_percentage_map.end(); ++map_iter) {
                    float intersection_percentage(map_iter->second);
                    if (intersection_percentage > best_intersection_percentage) {
                        best_intersection_percentage = intersection_percentage;
                        correct_detected_rect_pair = faces[map_iter->first];
                    }
                }

                //if enters here means there is good point intersection with a detected rect
                if (correct_detected_rect_pair.first.area() > 0) {
                    //have to clear the old points first!
                    track->current_points.clear();
                    //only add new points if they are a '1' in the status vector and within the correctly detected rect!!
                    //TODO: could also use the err vector with a float threshold
                    for (unsigned i = 0; i < status.size(); i++) {
                        if (static_cast<int>(status[i])) {
                            //TODO: keep track of how many missed detections per track - think about stopping the track if not
                            //detecting the face for a few frames in a row
                            //only keep if within the correct detected face rect
                            if (correct_detected_rect_pair.first.contains(new_points[i])) {
                                track->current_points.push_back(new_points[i]);
                                circle(frame_draw_pre_verified, new_points[i], 2, Scalar(255, 255, 255), cv::FILLED);
                            }
                            else {
                                circle(frame_draw_pre_verified, new_points[i], 2, Scalar(0, 0, 255), cv::FILLED);
                            }
                        }
                    }
                }
                else {
                    //if turning the additional three displays on then make sure they are killed when finished tracking or detecting!
                    //Display("current frame at lost face", frame);

                    //have to clear the old points first! - points are still in the new points vector
                    track->current_points.clear();

                    //add all of the points
                    for (unsigned i = 0; i < new_points.size(); i++) {
                        track->current_points.push_back(new_points[i]);
                        //draw the points as red
                        circle(frame_draw_pre_verified, new_points[i], 2, Scalar(0, 0, 255), cv::FILLED);
                    }

                    //draw a circle around the points
                    Point2f center;
                    float radius;
                    minEnclosingCircle(track->current_points, center, radius);
                    //radius is increased
                    circle(frame_draw_pre_verified, center, radius * 1.2, Scalar(255, 255, 255), 1, 8);

                    //get a bound rect using the current points
                    Rect face_rect = boundingRect(track->current_points);
                    //check to see how small the bounding rect is
                    if (face_rect.height < 32) //face_rect.width < 32 ||
                    {
                        //if face smaller than 32 pixels then we don't want to keep it
                        LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                                   << "] Face too small to track - killing track");
                        continue;
                    }

                    //increase size since the size was decreased when detecting the points
                    Rect upscaled_face = GetUpscaledFaceRect(face_rect);
                    rectangle(frame_draw_pre_verified, face_rect, Scalar(255, 255, 0));

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Getting template match");

                    //try template matching
                    //make sure the template is not out of bounds
                    AdjustRectToEdges(upscaled_face, gray);

                    //get first face - TODO: might not be the best face - should use the quality tool if available
                    //or could use the last face
                    Rect last_face_rect = Utils::ImageLocationToCvRect(track->face_track.frame_locations.rbegin()->second); // last element in map
                    Mat templ = prev_gray(last_face_rect);

                    //Display("last face", templ);

                    Rect match_rect = GetMatch(frame, gray, templ);
                    Mat new_frame_copy = frame.clone();

                    rectangle(new_frame_copy, match_rect, Scalar(255, 255, 255), 2);
                    //draw the previous face even though it was from the previous frame
                    rectangle(new_frame_copy, last_face_rect, Scalar(0, 255, 0), 2);

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Match rect area: "
                                                               << match_rect.area());

                    Rect match_intersection = match_rect & last_face_rect; //opencv allows for this operation
                    rectangle(new_frame_copy, match_intersection, Scalar(255, 0, 0), 2);

                    //Display("intersection", new_frame_copy);

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Finished getting match");

                    //look for a certain percentage of intersection
                    if (match_intersection.area() > 0) {
                        float intersection_rate = static_cast<float>(match_intersection.area()) /
                                                  static_cast<float>(last_face_rect.area());

                        LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Intersection rate: "
                                                                   << intersection_rate);

                        //was at 0.5 - should be much higher - don't want to be very permissive here - the template
                        //matching is not that good - TODO: need some sort of score could use openbr to do matching
                        if (intersection_rate < 0.7) {
                            continue;
                        }
                    }
                    else {
                        continue;
                    }

                    //NEED TO TRIM THE POINTS TO THE TEMPLATE MATCH RECT!!!
                    //some of the calc optical flow next points will start to get away!!
                    //TODO: make sure to set the actual rect to the template rect

                    //set the rescaled face that is used for the object detection to the template match
                    correct_detected_rect_pair.first = match_rect;

                    //set to true to make sure we don't redetect points in this case
                    track_recovered = true;

                    int points_within_template_match_rect = 0;
                    vector <Point2f> temp_points_copy(track->current_points);
                    //have to be cleared once again
                    track->current_points.clear();

                    //for(unsigned i=0; i < track->current_points.size(); i++)
                    for (unsigned i = 0; i < temp_points_copy.size(); i++) {
                        //USING the match intersection rect - might fix some issues with there is a transition in the video
                        //- might want to use the intserection for the actual face rect as well
                        if (match_rect.contains(temp_points_copy[i])) {
                            //TODO: need to check the count of these points - also need to kill the track if a bounding
                            //rect of these new points gets to be too small!!!
                            track->current_points.push_back(temp_points_copy[i]);
                            ++points_within_template_match_rect;

                        }
                    }
                }


                //TODO: does it seem necessary to do this again?
                //check corrected rect area again to check point percent
                if (correct_detected_rect_pair.first.area() > 0) {
                    //now can check the point percentage!
                    float current_point_percent = static_cast<float>(track->current_points.size()) /
                                                  static_cast<float>(track->init_point_count);

                    //TODO: probably need an intersection rate specific to the track continuation
                    //update current track info
                    track->current_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_percent = current_point_percent;

                    if (current_point_percent < min_point_percent) {
                        //lost too many of original points - kill track
                        //set something to continue onto the feature matching portion - no reason to kill the track here
                        LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                                   << "] Lost too many points, current percent: " << current_point_percent);
                        continue;
                    }
                }
            }

            bool redetect_feature_points = true;

            //if a face bounding box is now set and we can redetect feature points
            //if track is not recovered
            if(correct_detected_rect_pair.first.area() > 0 && redetect_feature_points &&
               !track_recovered && track->current_point_percent < min_redetect_point_perecent) {
                LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                           << "] Attempting to redetect feature points");
                vector <KeyPoint> keypoints;
                Mat mask = GetMask(gray, correct_detected_rect_pair.first);
                //search for keypoints in the frame using a mask
                feature_detector->detect(gray, keypoints, mask);

                //calcOpticalFlowPyrLK uses float points - no need to store the KeyPoint vector - convert
                track->current_points.clear(); //why not clear before
                KeyPoint::convert(keypoints, track->current_points);

                //min init point count should be different for each detector!
                //TODO: not sure if I want to kill the track here - just don't update the init point count and continue
                if(keypoints.size() < min_init_point_count)
                {
                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Not enough initial points: " <<
                                                               static_cast<int>(track->current_points.size()));

                    //set the init to min init point count because we are now below that
                    track->init_point_count = min_init_point_count;
                    track->current_point_count = static_cast<int>(track->current_points.size());

                    //now need to re-check the percentage - TODO: put this in a member function
                    float current_point_percent = static_cast<float>(track->current_points.size()) /
                                                  static_cast<float>(track->init_point_count);
                    //set track info for display
                    track->current_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_percent = current_point_percent;

                    if (current_point_percent < min_point_percent) {
                        //lost too many of original points - kill track
                        LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                                   << "] Lost too many points below min point percent, "
                                                                   << "current percent: " << current_point_percent);
                        continue;
                    }

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                               << "] Keeping track below min_init_point_count with current percent: "
                                                               << current_point_percent);
                }
                else {
                    //reset the init and current point count also!
                    track->init_point_count = static_cast<int>(track->current_points.size());
                    track->current_point_count = track->init_point_count;
                }

                //must update the last detection index!
                track->last_face_detected_index = frame_index;
            }

            //at this point if the correct detected rect has an area we can keep the track
            if (correct_detected_rect_pair.first.area() > 0) {
                rectangle(frame_draw_pre_verified, correct_detected_rect_pair.first, Scalar(0, 255, 0), 2);

                //don't want to store a face that isn't within the bounds of the image
                AdjustRectToEdges(correct_detected_rect_pair.first, gray);
                MPFImageLocation fd = Utils::CvRectToImageLocation(correct_detected_rect_pair.first);

                fd.confidence = static_cast<float>(correct_detected_rect_pair.second);
                //can finally store the MPFImageLocation
                track->face_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, fd));
                track->face_track.confidence = std::max(track->face_track.confidence, fd.confidence);
            }
            else {
                continue;
            }

            //if makes it here then we want to keep the track!!
            track->track_lost = false;
            //can also set the frame_draw to frame_draw_pre_verified Mat - TODO: should think of showing the pre verified Mat if the
            //track is lost
            frame_draw = frame_draw_pre_verified.clone();
        }

        //draw before killing bad tracks and adding new tracks!
        if (imshow_on) {
            vector <MPFVideoTrack> temp_tracks;
            //TODO: annoying to have to do this conversion - also losing point count info
            //TODO: need to store tracker point info in a different way to keep from having to do this conversion so many times!!
            for (vector<Track>::iterator cur_track = current_tracks.begin(); cur_track != current_tracks.end(); ++cur_track) {
                temp_tracks.push_back(cur_track->face_track);
            }
            vector <MPFImageLocation> empty_locations;
            Utils::DrawTracks(frame_draw, temp_tracks, empty_locations, static_cast<int>(saved_tracks.size()));
            Utils::DrawText(frame_draw, frame_index);

            //might not be a bad idea to update the display twice
            Display("Open Tracker", frame_draw);
        }

        //check if there is intersection between new objects and existing tracks
        //if not then add the new tracks
        for (unsigned i = 0; i < faces.size(); ++i) {
            int intersection_index = -1;
            if (!IsExistingTrackIntersection(faces[i].first, intersection_index)) {
                Track track_new;

                //set face detection
                Rect face(faces[i].first);
                AdjustRectToEdges(face, gray);

                float first_face_confidence = static_cast<float>(faces[i].second);

                bool use_face = false;
                if (first_face_confidence > min_initial_confidence) {
                    use_face = true;
                }
                else {
                    if(imshow_on) {
                        //draw the track detection red for bad quality
                        rectangle(frame_draw, face, Scalar(0, 0, 255), 3);

                        Display("Open Tracker", frame_draw);
                    }

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name
                                                               << "] Detected face does not meet initial quality: " << first_face_confidence);
                }

                if (use_face) {
                    //if the face meets quality or we don't care about quality
                    //the keypoints can now be detected

                    vector <KeyPoint> keypoints;
                    Mat mask = GetMask(gray, faces[i].first);
                    //search for keypoints in the frame using a mask
                    feature_detector->detect(gray, keypoints, mask);

                    //min init point count should be different for each detector!
                    if(keypoints.size() < min_init_point_count)
                    {
                        LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Not enough initial points: "
                                                                   << static_cast<int>(keypoints.size()));

                        if(imshow_on) {
                            //draw the track detection red for bad point count
                            rectangle(frame_draw, face, Scalar(0, 0, 255), 3);

                            Display("Open Tracker", frame_draw);
                        }

                        continue;
                    }

                    //set first keypoints and mat
                    track_new.first_detected_keypoints = keypoints;
                    track_new.first_gray_frame = gray.clone();

                    //calcOpticalFlowPyrLK uses float points - no need to store the KeyPoint vector - convert
                    KeyPoint::convert(keypoints, track_new.current_points);

                    //drawing new points and detection rectangle
                    //image will already contain previously drawn objects
                    if(imshow_on) {
                        for(unsigned k=0; k<track_new.current_points.size(); k++) //TODO: could also use the err vector (from calcOpticalFlowPyrLK) with a float threshold
                        {
                            circle(frame_draw, track_new.current_points[k], 2, Scalar(0, 255, 255), cv::FILLED);
                        }

                        Display("Open Tracker", frame_draw);
                    }

                    //set start frame and initial point count
                    track_new.face_track.start_frame = frame_index;
                    //set first face detection index
                    track_new.last_face_detected_index = frame_index;
                    track_new.init_point_count = static_cast<int>(track_new.current_points.size());

                    //set face detection
                    Rect face(faces[i].first);
                    AdjustRectToEdges(face, gray);
                    MPFImageLocation first_face_detection(face.x, face.y, face.width, face.height);
                    //first_face_confidence is already a float value
                    first_face_detection.confidence = first_face_confidence;

                    //add the first detection
                    track_new.face_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, first_face_detection));
                    track_new.face_track.confidence = std::max(track_new.face_track.confidence,
                                                               first_face_detection.confidence);
                    //add the new track
                    current_tracks.push_back(track_new);

                    LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Creating new track");
                }


            }

        }

        vector <Track> tracks_to_keep;
        for(vector<Track>::iterator track = current_tracks.begin(); track != current_tracks.end(); ++track)
        {
            if(track->track_lost)
            {
                LOG4CXX_TRACE(OpenFaceDetectionLogger, "[" << job.job_name << "] Killing track");

                //did not pass the rules to continue this frame_index, it ended on the previous index
                track->face_track.stop_frame = frame_index - 1;

                //only saving tracks lasting more than 1 frame to eliminate badly started tracks
                if (track->face_track.stop_frame - track->face_track.start_frame > 1) {
                    saved_tracks.push_back(*track);
                }
            }
            else {
                tracks_to_keep.push_back(*track);
            }
        }

        //now clear the current tracks and add the tracks to keep
        current_tracks.clear();
        for (vector<Track>::iterator track = tracks_to_keep.begin(); track != tracks_to_keep.end(); track++) {
            current_tracks.push_back(*track);
        }

        //set previous frame
        prev_gray = gray.clone();
        //swap points
        for (vector<Track>::iterator it = current_tracks.begin(); it != current_tracks.end(); it++) {
            swap(it->current_points, it->previous_points);
        }

        ++frame_index;
    }

    CloseAnyOpenTracks(video_capture.GetFrameCount() - 1);

    vector<MPFVideoTrack> tracks;
    //set tracks reference!
    for (unsigned int i = 0; i < saved_tracks.size(); i++) {
        tracks.push_back(saved_tracks[i].face_track);
    }

    //clear any internal structures that could carry over before the destructor is called
    //these can be cleared - the data has been moved to tracks
    current_tracks.clear();
    saved_tracks.clear();

    LOG4CXX_INFO(OpenFaceDetectionLogger, "[" << job.job_name << "] Processing complete. Found "
                                              << static_cast<int>(tracks.size()) << " tracks.");
    CloseWindows();

    if (verbosity > 0) {
        //now print tracks if available
        if(!tracks.empty())
        {
            for(unsigned int i=0; i<tracks.size(); i++)
            {
                LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Track index: " << i);
                LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Track start index: " << tracks[i] .start_frame);
                LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Track end index: " << tracks[i] .stop_frame);

                for (map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it)
                {
                    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Frame num: " << it->first);
                    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Bounding rect: (" << it->second .x_left_upper << ","
                                                               <<  it->second.y_left_upper << "," << it->second.width << "," << it->second.height << ")");
                    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Confidence: " << it->second.confidence);
                }
            }
        }
        else
        {
            LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] No tracks found");
        }
    }

    return tracks;
}


vector<MPFImageLocation> OcvFaceDetection::GetDetections(const MPFImageJob &job) {
    try {
        SetDefaultParameters();
        SetReadConfigParameters();
        GetPropertySettings(job.job_properties);

        MPFImageReader imreader(job);
        cv::Mat image_data(imreader.GetImage());

        vector<MPFImageLocation> locations = GetDetectionsFromImageData(job, image_data);

        for (auto &location : locations) {
            imreader.ReverseTransform(location);
        }
        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, OpenFaceDetectionLogger);
    }
}

vector<MPFImageLocation> OcvFaceDetection::GetDetectionsFromImageData(
        const MPFImageJob &job, cv::Mat &image_data) {

    int frame_width = 0;
    int frame_height = 0;

    /**************************/
    /* Read and Submit Image */
    /**************************/

    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Getting detections");


    cv::Mat image_gray = Utils::ConvertToGray(image_data);

    frame_width = image_data.cols;
    frame_height = image_data.rows;
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Frame_width = " << frame_width);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Frame_height = " << frame_height);

    vector<pair<cv::Rect,int>> face_rects = ocv_detection.DetectFaces(image_gray);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Number of faces detected = " << face_rects.size());

    vector<MPFImageLocation> locations;
    for (unsigned int j = 0; j < face_rects.size(); j++) {
        cv::Rect face = face_rects[j].first;

        //pass face by ref
        AdjustRectToEdges(face, image_data);

        float confidence = static_cast<float>(face_rects[j].second);

        MPFImageLocation location = Utils::CvRectToImageLocation(face, confidence);

        locations.push_back(location);
    }

    if (verbosity) {
        // log the detections
        for (unsigned int i = 0; i < locations.size(); i++) {
            LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job.job_name << "] Detection # " << i);
            LogDetection(locations[i], job.job_name);
        }
    }

    if (verbosity > 0) {
        //    Draw a rectangle onto the input image for each detection
        if (imshow_on) {
            cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
            imshow("original image", image_data);
            cv::waitKey(5);
        }
        for (unsigned int i = 0; i < locations.size(); i++) {
            cv::Rect object(locations[i].x_left_upper,
                            locations[i].y_left_upper,
                            locations[i].width,
                            locations[i].height);
            rectangle(image_data, object, CV_RGB(0, 0, 0), 2);
        }
        if (imshow_on) {
            cv::namedWindow("new image", cv::WINDOW_AUTOSIZE);
            imshow("new image", image_data);
            //0 waits indefinitely for input, which could cause problems when run as a component
            //cv::waitKey(0);
            cv::waitKey(5);
        }
        QString imstr(job.data_uri.c_str());
        QFile f(imstr);
        QFileInfo fileInfo(f);
        QString fstr(fileInfo.fileName());
        string outfile_name = "output_" + fstr.toStdString();
        try {
            imwrite(outfile_name, image_data);
        }
        catch (runtime_error &ex) {
            CloseWindows();
            throw MPFDetectionException(
                    MPF_FILE_WRITE_ERROR,
                    std::string("Exception writing image output file: ") + ex.what());
        }
    }

    LOG4CXX_INFO(OpenFaceDetectionLogger, "[" << job.job_name << "] Processing complete. Found "
                                              << static_cast<int>(locations.size()) << " detections.");

    CloseWindows();


    return locations;
}


void OcvFaceDetection::LogDetection(const MPFImageLocation& face, const string& job_name)
{
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job_name << "] XLeftUpper: " << face.x_left_upper);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job_name << "] YLeftUpper: " << face.y_left_upper);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job_name << "] Width:      " << face.width);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job_name << "] Height:     " << face.height);
    LOG4CXX_DEBUG(OpenFaceDetectionLogger, "[" << job_name << "] Confidence: " << face.confidence);
}


void OcvFaceDetection::CloseWindows() {
    if(imshow_on) {
        destroyAllWindows();
        waitKey(5); //waitKey might need to be called to actually kill the windows?
    }
}


MPF_COMPONENT_CREATOR(OcvFaceDetection);
MPF_COMPONENT_DELETER();
