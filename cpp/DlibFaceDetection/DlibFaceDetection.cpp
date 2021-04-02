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

#include "DlibFaceDetection.h"

#include <limits.h>
#include <unistd.h>
#include <fstream>

#include <QFile>
#include <QFileInfo>


#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>

#include "MPFImageReader.h"
#include "MPFSimpleConfigLoader.h"
#include "detectionComponentUtils.h"

using std::string;
using std::vector;
using std::map;
using std::size_t;
using std::begin;
using std::end;

using cv::Mat;
using cv::Rect;
using cv::Scalar;
using cv::imshow;
using cv::namedWindow;
using cv::destroyAllWindows;
using cv::waitKey;
using cv::pyrUp;
using cv::pyrDown;

using dlib::correlation_tracker;
using dlib::rectangle;
using dlib::rect_detection;

using namespace std;
using namespace MPF::COMPONENT;


string DlibFaceDetection::GetDetectionType() {
    return "FACE";
}

void DlibFaceDetection::SetModes(bool display_window, bool print_debug_info) {
    imshow_on = display_window;

    if (print_debug_info && logger_ != NULL) {
        logger_->setLevel(log4cxx::Level::getDebug());
    }
}

bool DlibFaceDetection::Init() {

    // Determine where the executable is running
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/DlibFaceDetection";
    string config_path = plugin_path + "/config";

    logger_ = log4cxx::Logger::getLogger("DlibFaceDetection");

    // uncommenting this line will cause all log statements to go to the detection log
    // instead of the dlib_face_detection log
    //    if (log4cxx::Logger::getRootLogger()->getAllAppenders().size() == 0)

    LOG4CXX_DEBUG(logger_, "Plugin path: " << plugin_path);

    LOG4CXX_INFO(logger_, "Initializing Dlib Face");

    dlib_face_detector = dlib::get_frontal_face_detector();

    SetDefaultParameters();

    //once this is done - parameters will be set and SetReadConfigParameters() can be called again to revert back
    //to the params read at initialization

    string config_params_path = config_path + "/mpfDlibFaceDetection.ini";
    int rc = LoadConfig(config_params_path, parameters);
    if (rc) {
        LOG4CXX_ERROR(logger_, "Could not parse config file: " << config_params_path);
        return (false);
    }

    SetReadConfigParameters();

    return true;
}

bool DlibFaceDetection::Close() {
    return true;
}

/*
 * Called during Init
 */
void DlibFaceDetection::SetDefaultParameters() {

    verbosity = 0;
    imshow_on = false;

    //min dlib object detection confidence
    //needed to start a new track
    min_detection_confidence = 0.1;

    //the maximum allowable overlap
    //rate between a new detection rect and
    //any existing track rects
    max_intersection_overlap_pct = 0.2f;

    //the amount of frame locations required
    //to save a track
    min_track_length = 3;

    //the minimum amount of similary required
    //by a detection to be matched with an
    //existing track
    min_track_object_similarity_value = 0.6f;

    //the minimum amount of correlation
    //between frames needed to continue tracking
    min_update_correlation = 6.5;

    //NOT ADDED TO THE CONFIG
    //this is the bounding box grow rate
    // that is used to grow the detection
    // rect before using it as a guess
    // rect for the tracker
    bb_grow_rate = 0.08f;
}

/*
 * Called during Init
 */
void DlibFaceDetection::SetReadConfigParameters() {
    //make sure none of the parameters are missed in the config file - double check

    if(parameters.contains("VERBOSE")) {
        //right now only accepting a VERBOSITY of 1 and just checking for > 0, may need to adjust later
        //if VERBOSITY 1 set the log level to DEBUG
        //if VERBOSITY set to 2, think about using TRACE
        verbosity = parameters["VERBOSE"].toInt();
        if (verbosity > 0) {
            logger_->setLevel(log4cxx::Level::getDebug());
        }
    }

    if(parameters.contains("IMSHOW_ON")) {
        imshow_on = (parameters["IMSHOW_ON"].toInt() > 0);
    }

    if(parameters.contains("MIN_DETECTION_CONFIDENCE")) {
        min_detection_confidence = parameters["MIN_DETECTION_CONFIDENCE"].toDouble();
    }

    if(parameters.contains("MAX_INTERSECTION_OVERLAP_AREA_PCT")) {
        max_intersection_overlap_pct = parameters["MAX_INTERSECTION_OVERLAP_AREA_PCT"].toFloat();
    }

    if(parameters.contains("MIN_TRACK_LENGTH")) {
        min_track_length = parameters["MIN_TRACK_LENGTH"].toInt();
    }

    if(parameters.contains("MIN_TRACK_OBJECT_SIMILARITY_VALUE")) {
        min_track_object_similarity_value = parameters["MIN_TRACK_OBJECT_SIMILARITY_VALUE"].toFloat();
    }

    if(parameters.contains("MIN_UPDATE_CORRELATION")) {
        min_update_correlation = parameters["MIN_UPDATE_CORRELATION"].toDouble();
    }
}

/*
 * This function reads a property value map and adjusts the settings for this component
 * and is called at the beginning of detection
 */
void DlibFaceDetection::GetPropertySettings(const map <string, string> &algorithm_properties) {
    string property;
    string str_value;
    int ivalue;
    float fvalue;
    for (map<string, string>::const_iterator imap = algorithm_properties.begin();
         imap != algorithm_properties.end(); imap++) {
        property = imap->first;
        str_value = imap->second;

        //TODO: could restrict some of the parameter ranges here and log like PP!
        if (property == "VERBOSE") { //INT
            verbosity = atoi(str_value.c_str());
        }
        else if (property == "MIN_DETECTION_CONFIDENCE") { //DOUBLE
            min_detection_confidence = static_cast<double>(atof(str_value.c_str()));
        }
        else if (property == "MAX_INTERSECTION_OVERLAP_AREA_PCT") { //FLOAT
            max_intersection_overlap_pct = atof(str_value.c_str());
        }
        else if (property == "MIN_TRACK_OBJECT_SIMILARITY_VALUE") { //FLOAT
            min_track_object_similarity_value = atof(str_value.c_str());
        }
        else if (property == "MIN_UPDATE_CORRELATION") { //DOUBLE
            min_update_correlation = static_cast<double>(atof(str_value.c_str()));
        }
    }
    return;
}

/*
 * Determine how similar the current_track rectangle (last position) is to the new_rect
 */
float DlibFaceDetection::GetTrackObjectSimilarity(const DlibTrack &current_track, const rectangle &new_rect) {
    rectangle intersect_rect = current_track.correlation_tracker.get_position().intersect(new_rect);

    float intersect_ratio = intersect_rect.area()/(static_cast<float>(current_track.correlation_tracker.get_position().area()));

    //determine if the area of the new detected rect is close to the last correlation position
    //float area_similarity_ratio = new_rect.area()/(static_cast<float>(current_track.correlation_tracker.get_position().area()));
    //TODO: consider using area_similarity_ratio combined with intersect_ratio
    float similarity = intersect_ratio;

    return similarity;
}

/*
 * This will determine if the last position in the track (current_track) is similar to the rectangle (new_rect)
 */
bool DlibFaceDetection::IsObjectSimilar(const DlibTrack &current_track, const rectangle &new_rect) {
    float similarity = GetTrackObjectSimilarity(current_track, new_rect);
    if(similarity >= min_track_object_similarity_value) {
        return true;
    }
    return false;
}

/*
 * Search through a set of rectangles to find the most similar overlapping object
 * when compared to the current track's (current_track) last position
 * Returns the index of the most similar overlapping object in next_detected_objects
 * Return -1 if nothing found
 */
int DlibFaceDetection::GetMostSimilarOverlappingObject(const DlibTrack &current_track,
                                                       const vector<rect_detection> &next_detected_objects) {

    int most_similar_index = -1;
    float best_similarity_value = 0.0f;
    for (size_t i = 0; i != next_detected_objects.size(); ++i) {
        //similarity will be 0 if not overlapping
        float track_object_similarity = GetTrackObjectSimilarity(current_track, next_detected_objects[i].rect);
        if(track_object_similarity >= min_track_object_similarity_value &&
           track_object_similarity > best_similarity_value) {
            best_similarity_value = track_object_similarity;
            //set index - will not overflow in this case
            most_similar_index = static_cast<int>(i);
        }
    }

    return most_similar_index;
}

/*
 * An object can be found as not similar (IsObjectSimilar = false) but it may still be overlapping an existing track too much
 * and should not be used to create a new track
 */
bool DlibFaceDetection::IsValidNewObject(const DlibTrack &current_track, const rectangle &new_rect) {

    rectangle intersect_rect = current_track.correlation_tracker.get_position().intersect(new_rect);

    float intersect_ratio = intersect_rect.area()/(static_cast<float>(current_track.correlation_tracker.get_position().area()));

    //TODO: could allow some overlap and check if "intersect_ratio <= MAX_INTERSECTION_OVERLAP_AREA_PCT"
    //rather than not allowing any intersection
    if(intersect_ratio > 0.0f) {
        //not a valid object
        return false;
    }
    return true;
}

void DlibFaceDetection::CloseAnyOpenTracks() {
    if (!current_tracks.empty()) {
        //need to stop all current tracks!
        for(auto &current_track : current_tracks) {

            //should never happen - but ignoring the track if stop frame already modified or
            //there are less than MIN_TRACK_LENGTH frame locations
            if (current_track.mpf_video_track.stop_frame != -1
                || current_track.mpf_video_track.frame_locations.size() < min_track_length) {
                continue;
            }

            //track is still going at end index
            //set the stop_frame for this track!!!
            //get last frame location entry
            int last_used_frame_location_index = current_track.mpf_video_track.frame_locations.rbegin()->first;
            current_track.mpf_video_track.stop_frame = last_used_frame_location_index;

            //now the track can be saved
            saved_tracks.push_back(current_track);
        }
    }
}

void DlibFaceDetection::GrowRect(rectangle &rect) {

    int width_adjust = floor( (rect.width() * bb_grow_rate) / 2.0f);
    int height_adjust = floor( (rect.height() * bb_grow_rate) / 2.0f);

    rect.set_left((rect.left() - width_adjust));
    rect.set_top((rect.top() - height_adjust));

    rect.set_right((rect.right() + width_adjust));
    rect.set_bottom((rect.bottom() + height_adjust));

    //the width and height of the rect are calculated
    // based on the corners
}

void DlibFaceDetection::AdjustRectToEdgesDlib(rectangle &rect, const Mat &src) {
    if (!src.empty()) {
        //check corners and edges and resize appropriately!

        //modifying the referenced rect
        //subtracting 1 since indexes are 0 based, if image is 256x256 there is will be 256 rows and cols,
        //but the values will range from 0 to 255
        int x_max = (src.cols - 1);
        int y_max = (src.rows - 1);

        //x left
        if(rect.left() < 0) {
            rect.set_left(0);
        }

        //x right
        if(rect.right() > x_max) {
            rect.set_right(x_max);
        }

        //y top
        if(rect.top() < 0) {
            rect.set_top(0);
        }

        //y bottom
        if(rect.bottom() > y_max) {
            rect.set_bottom(y_max);
        }

        //width and height are calculated using the left,
        // right, top, and bottom values when requested
    }
}

void DlibFaceDetection::DlibRectToMPFImageLocation(const rectangle &object_rect, float object_detection_confidence,
                                                   MPFImageLocation &mpf_object_detection) {
    mpf_object_detection = MPFImageLocation(object_rect.left(), object_rect.top(),
                                            object_rect.width(), object_rect.height(), object_detection_confidence);
}

void DlibFaceDetection::UpdateTracks(const dlib::cv_image<dlib::uint8> &next_frame_gray, const Mat &next_frame_gray_mat,
                                     vector<rect_detection> &next_detected_objects, int frame_index) {

    //loop through existing tracks locating the most similar newly detected object (from next_detected_objects)
    //remove any newly detected objects that are used
    auto current_track = begin(current_tracks);
    while (current_track != end(current_tracks)) {

        //remember that "current_track->updated" is not the same as the correlation_tracker update - that happens each time
        //UpdateTracks is called. If updated is true then the track location will have an updated confidence value and if false
        //the value will be 0.0f
        //back to false before seeing if there is a new rect that can be used for updating
        current_track->updated = false;

        //GetMostSimilarOverlappingObject can be used without checking all tracks to see if a track might share more similary to one of the objects
        // because the detections should not overlap and will require a high percentage of overlap to even be considered similar
        int most_similar_index = GetMostSimilarOverlappingObject(*current_track, next_detected_objects);

        //tracker update confidence
        double update_conf = -1.0f;
        //object detection confidence
        double object_location_conf = 0.0f;

        //GetMostSimilarOverlappingObject returns -1 if it cannot find a rect to use
        if(most_similar_index != -1) {
            rect_detection valid_most_similar_object = next_detected_objects.at(most_similar_index);
            rectangle valid_most_similar_object_rect = valid_most_similar_object.rect;

            //duplicate the rect to modify
            rectangle rect_to_grow(valid_most_similar_object_rect.tl_corner(), valid_most_similar_object_rect.br_corner());

            //grow the rect for help in guessing
            // the new track position
            GrowRect(rect_to_grow);
            //adjust to image bounds
            AdjustRectToEdgesDlib(rect_to_grow, next_frame_gray_mat);

            //now try to update with grown rect
            update_conf = current_track->correlation_tracker.update(next_frame_gray, rect_to_grow);
            current_track->updated = true;
            current_track->frames_since_last_detection = 0;

            object_location_conf = valid_most_similar_object.detection_confidence;

            //with the valid_most_similar_object_rect from next_detected_objects being used, erase it to reduce looping later
            next_detected_objects.erase(next_detected_objects.begin() + most_similar_index);
        } else {
            //update without a guess
            update_conf = current_track->correlation_tracker.update(next_frame_gray);
        }

        if (update_conf >= min_update_correlation) {
            MPFImageLocation mpf_object_detection;
            DlibRectToMPFImageLocation(current_track->correlation_tracker.get_position(), object_location_conf, mpf_object_detection);
            current_track->mpf_video_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, mpf_object_detection));
            current_track->mpf_video_track.confidence = std::max(current_track->mpf_video_track.confidence,
                                                                 mpf_object_detection.confidence);
            //next track
            ++current_track;
        } else {
            //stop the track, save if meets requirements, and delete from current tracks

            if(current_track->mpf_video_track.frame_locations.size() > min_track_length) {
                //since the frame interval can be adjusted it makes sense to grab the index from the last frame location
                current_track->mpf_video_track.stop_frame = current_track->mpf_video_track.frame_locations.rbegin()->first;
                saved_tracks.push_back(*current_track);
            }

            current_track = current_tracks.erase(current_track);
        }
    }

    auto detected_object = begin(next_detected_objects);
    while (detected_object != end(next_detected_objects)) {

        rectangle detected_object_rect = detected_object->rect;

        //need to iterate if not used
        bool use_detected_object = true;

        for (const auto &existing_track : current_tracks) {
            if(!IsValidNewObject(existing_track, detected_object_rect)) {
                use_detected_object = false;
                break;
            }
        }

        if(use_detected_object) {
            //create new track track!
            DlibTrack new_dlib_track;
            new_dlib_track.mpf_video_track.start_frame = frame_index;
            new_dlib_track.correlation_tracker.start_track(next_frame_gray, detected_object_rect);

            MPFImageLocation first_mpf_object_detection;
            DlibRectToMPFImageLocation(detected_object_rect, detected_object->detection_confidence, first_mpf_object_detection);
            new_dlib_track.mpf_video_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_index, first_mpf_object_detection));
            new_dlib_track.mpf_video_track.confidence = std::max(new_dlib_track.mpf_video_track.confidence,
                                                                 first_mpf_object_detection.confidence);

            current_tracks.push_back(new_dlib_track);

            //delete this new detect object
            detected_object = next_detected_objects.erase(detected_object);
        } else {
            ++detected_object;
        }
    }
}

vector<MPFVideoTrack> DlibFaceDetection::GetDetectionsFromVideoCapture(
        const MPFVideoJob &job, MPFVideoCapture &video_capture) {

    int total_frames = video_capture.GetFrameCount();
    LOG4CXX_INFO(logger_, "[" << job.job_name << "] Total video frames: " << total_frames);

    int frame_index = 0;

    Mat frame, gray;

    if(imshow_on) {
        namedWindow("Tracker Window", cv::WINDOW_AUTOSIZE );
    }

    while (video_capture.Read(frame)) {


        //Convert to grayscale - make sure not to duplicate this step in detection
        gray = Utils::ConvertToGray(frame);

        //look for new objects
        vector<rect_detection> objects_detected = DetectFacesDlib(gray);

        dlib::cv_image<dlib::uint8> dlib_img(gray);
        UpdateTracks(dlib_img, gray, objects_detected, frame_index);

        if(imshow_on) {
            //can draw on frame because the detection step is complete

            for(auto &current_track : current_tracks) {
                //get last object location
                MPFImageLocation last_mpf_object_location = current_track.mpf_video_track.frame_locations.rbegin()->second;

                Rect cv_rect(last_mpf_object_location.x_left_upper, last_mpf_object_location.y_left_upper,
                             last_mpf_object_location.width, last_mpf_object_location.height);

                if(current_track.updated) {
                    cv::rectangle(frame, cv_rect, Scalar(255,255,0));
                } else {
                    cv::rectangle(frame, cv_rect, Scalar(0,0,255));
                }
            }

            imshow("Tracker Window", frame);
            waitKey(5);
        }

        ++frame_index;
    }
    CloseAnyOpenTracks();

    vector<MPFVideoTrack> tracks;
    //set tracks reference!
    for (unsigned int i = 0; i < saved_tracks.size(); i++) {
        tracks.push_back(saved_tracks[i].mpf_video_track);
    }

    //clear any internal structures that could carry over before the destructor is called
    //these can be cleared - the data has been moved to tracks
    current_tracks.clear();
    saved_tracks.clear();

    LOG4CXX_INFO(logger_, "[" << job.job_name << "] Processing complete. Found "
                              << static_cast<int>(tracks.size()) << " tracks.");
    CloseWindows();

    if (verbosity > 0) {
        //now print tracks if available
        if(!tracks.empty())
        {
            for(unsigned int i=0; i<tracks.size(); i++)
            {
                LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Track index: " << i);
                LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Track start index: " << tracks[i].start_frame);
                LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Track end index: " << tracks[i].stop_frame);

                for (map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it)
                {
                    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Frame num: " << it->first);
                    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Bounding rect: (" << it->second.x_left_upper << ","
                                               <<  it->second.y_left_upper << "," << it->second.width << "," << it->second.height <<
                                               ")");
                    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Confidence: " << it->second.confidence);
                }
            }
        } else {
            LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] No tracks found");
        }
    }

    return tracks;
}


vector<MPFVideoTrack> DlibFaceDetection::GetDetections(const MPFVideoJob &job) {
    try {
        //set params to default and what was originally loaded in the .ini
        SetDefaultParameters();
        SetReadConfigParameters();

        //configure params
        //algorithm_properties
        /* Use the algorithm properties map to adjust the settings, if not empty */
        GetPropertySettings(job.job_properties);

        MPFVideoCapture video_capture(job, true, true);

        vector<MPFVideoTrack> tracks = GetDetectionsFromVideoCapture(job, video_capture);
        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }
        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}



// private - IMAGE
vector<MPFImageLocation> DlibFaceDetection::GetDetectionsFromImageData(const MPFImageJob &job, Mat &image) {

    /**************************/
    /* Read and Submit Image */
    /**************************/

    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Getting detections");


    cv::Mat image_gray = Utils::ConvertToGray(image);

    int frame_width = image.cols;
    int frame_height = image.rows;
    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Frame_width = " << frame_width);
    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Frame_height = " << frame_height);

    vector<rect_detection> object_detections = DetectFacesDlib(image_gray);

    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Number of faces detected = " << object_detections.size());

    vector<MPFImageLocation> locations;
    for(const auto &object_detection : object_detections) {
        float confidence = static_cast<float>(object_detection.detection_confidence);

        rectangle object_rect = object_detection.rect;

        MPFImageLocation mpf_object_detection(object_rect.left(), object_rect.top(),
                                              object_rect.width(), object_rect.height(), confidence);

        locations.push_back(mpf_object_detection);
    }

    if (verbosity > 0) {
        // log the detections
        for (unsigned int i = 0; i < locations.size(); i++) {
            LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Detection # " << i);
            LogDetection(locations[i], job.job_name);
        }
    }

    if (verbosity > 0) {
        //    Draw a rectangle onto the input image for each detection
        if (imshow_on) {
            namedWindow("original image", cv::WINDOW_AUTOSIZE);
            imshow("original image", image);
            waitKey(5);
        }
        for (unsigned int i = 0; i < locations.size(); i++) {
            Rect object(locations[i].x_left_upper,
                        locations[i].y_left_upper,
                        locations[i].width,
                        locations[i].height);
            cv::rectangle(image, object, CV_RGB(0, 0, 0), 2);
        }
        if (imshow_on) {
            namedWindow("new image", cv::WINDOW_AUTOSIZE);
            imshow("new image", image);
            //0 waits indefinitely for input, which could cause problems when run as a component
            waitKey(5);
        }
        QString imstr(job.data_uri.c_str());
        QFile f(imstr);
        QFileInfo fileInfo(f);
        QString fstr(fileInfo.fileName());
        string outfile_name = "output_" + fstr.toStdString();
        try {
            imwrite(outfile_name, image);
        }
        catch (runtime_error &ex) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name << "] Exception writing image output file: " << ex.what());
            CloseWindows();
            throw MPFDetectionException(MPF_OTHER_DETECTION_ERROR_TYPE,
                    std::string("Exception writing image output file: ") + ex.what());
        }
    }

    LOG4CXX_INFO(logger_, "[" << job.job_name << "] Processing complete. Found "
                              << static_cast<int>(locations.size()) << " detections.");

    CloseWindows();
    return locations;
}


vector<MPFImageLocation> DlibFaceDetection::GetDetections(const MPFImageJob &job) {
    try {
        //set params to default and what was originally loaded in the .ini
        SetDefaultParameters();
        SetReadConfigParameters();

        //configure params
        //algorithm_properties
        /* Use the algorithm properties map to adjust the settings, if not empty */
        GetPropertySettings(job.job_properties);

        MPFImageReader image_reader(job);
        cv::Mat image = image_reader.GetImage();

        vector<MPFImageLocation> locations = GetDetectionsFromImageData(job, image);
        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }

}



vector<rect_detection> DlibFaceDetection::DetectFacesDlib(const Mat &frame_gray) {

    //clone for pyramid up
    Mat frame_gray_clone = frame_gray.clone();
    equalizeHist( frame_gray_clone, frame_gray_clone);

    //the beginning of this is from a dlib detection example
    // "Make the image bigger by a factor of two.  This is useful since
    // the face detector looks for faces that are about 80 by 80 pixels
    // or larger.  Therefore, if you want to find faces that are smaller
    // than that then you need to upsample the image as we do here by
    // calling pyramid_up().  So this will allow it to detect faces that
    // are at least 40 by 40 pixels in size.  We could call pyramid_up()
    // again to find even smaller faces, but note that every time we
    // upsample the image we make the detector run slower since it must
    // process a larger image."
    //will have to pyrdown the detection values
    //From OpenCV documentation - By default, size of the output image is computed as Size(src.cols*2, (src.rows*2), but in any case, the following conditions should be satisfied:
    //Note that this also blurs the image as well
    pyrUp(frame_gray_clone, frame_gray_clone);

    dlib::cv_image<dlib::uint8> cimg(frame_gray_clone);

    vector<rect_detection> object_detections;
    dlib_face_detector(cimg, object_detections, min_detection_confidence);

    Mat frame_gray_clone_down;
    string window_name_up = "Detected dlib Faces pyrUp";
    string window_name_down = "Detected dlib Faces pyrDown";
    if(imshow_on) {
        frame_gray_clone_down = frame_gray.clone();
        namedWindow(window_name_up, cv::WINDOW_AUTOSIZE);
        namedWindow(window_name_down, cv::WINDOW_AUTOSIZE);
    }

    //need to pyramid down the object_detections rectangles
    for(auto &object_detection : object_detections) {

        if(imshow_on) {
            cv::Rect rect(object_detection.rect.tl_corner().x(), object_detection.rect.tl_corner().y(),
                          object_detection.rect.width(), object_detection.rect.height());
            cv::rectangle(frame_gray_clone, rect, cv::Scalar(255,0,0));
        }

        //dlib uses long, yes long, for its rectangle x, y, h, w
        //divide everything by 2 after the pyrUp
        //floor bottom left x and top right y and ceil of bottom left y and top right x (round up the rectangle)
        //cv mat reads from the top left (0,0)
        int left = floor(((float) object_detection.rect.tl_corner().x()) / 2.0f);
        int top = floor(((float) object_detection.rect.tl_corner().y()) / 2.0f);
        int right = ceil(((float) object_detection.rect.br_corner().x()) / 2.0f);
        int bottom = ceil(((float) object_detection.rect.br_corner().y()) / 2.0f);

        //set to rect adjusted after pyr up
        rectangle rect_to_adjust(left, top, right, bottom);

        //pass ref to update
        AdjustRectToEdgesDlib(rect_to_adjust, frame_gray);

        //point to the pyr up fixed and adjusted rect
        object_detection.rect = rect_to_adjust;

        if(imshow_on) {
            //see what the adjusted rect looks like on a pyDown'd Mat
            cv::Rect rect(object_detection.rect.tl_corner().x(), object_detection.rect.tl_corner().y(),
                          object_detection.rect.width(), object_detection.rect.height());
            cv::rectangle(frame_gray_clone_down, rect, cv::Scalar(255,0,0));
        }
    }

    if(imshow_on) {
        imshow(window_name_up, frame_gray_clone);
        imshow(window_name_down, frame_gray_clone_down);
        waitKey(5);
    }

    return object_detections;
}

void DlibFaceDetection::LogDetection(const MPFImageLocation& face, const string& job_name) {
    LOG4CXX_DEBUG(logger_, "[" << job_name << "] XLeftUpper: " << face.x_left_upper);
    LOG4CXX_DEBUG(logger_, "[" << job_name << "] YLeftUpper: " << face.y_left_upper);
    LOG4CXX_DEBUG(logger_, "[" << job_name << "] Width:      " << face.width);
    LOG4CXX_DEBUG(logger_, "[" << job_name << "] Height:     " << face.height);
    LOG4CXX_DEBUG(logger_, "[" << job_name << "] Confidence: " << face.confidence);
}

void DlibFaceDetection::CloseWindows() {
    if(imshow_on) {
        destroyAllWindows();
    }
}



MPF_COMPONENT_CREATOR(DlibFaceDetection);
MPF_COMPONENT_DELETER();
