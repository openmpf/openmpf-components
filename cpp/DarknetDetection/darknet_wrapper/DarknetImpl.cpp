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


#include <fstream>
#include <iostream>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <unordered_map>

#ifdef GPU
#include <cuda_runtime_api.h>
#endif

#include <opencv2/imgproc.hpp>

#include <MPFDetectionComponent.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>
#include <detectionComponentUtils.h>
#include <Utils.h>
#include <MPFDetectionException.h>
#include <MPFInvalidPropertyException.h>

#include "Trackers.h"
#include "DarknetImpl.h"


using namespace MPF::COMPONENT;


namespace DarknetHelpers {

    ProbHolder::ProbHolder(int output_layer_size, int num_classes)
        // The make_probs function from the Darknet library allocates num_classes + 1 floats per row.
        // There is no documentation so it isn't clear why it uses num_classes + 1 instead of just num_classes.
        : mat_size_(output_layer_size * (num_classes + 1))
        , prob_mat_(new float[mat_size_]())
        , prob_row_ptrs_(new float*[output_layer_size])
    {
        for (int i = 0; i < output_layer_size; i++) {
            prob_row_ptrs_[i] = &prob_mat_[i * (num_classes + 1)];
        }
    }

    float** ProbHolder::Get() {
        return prob_row_ptrs_.get();
    }

    float* ProbHolder::operator[](size_t box_idx) {
        return prob_row_ptrs_[box_idx];
    }

    void ProbHolder::Clear() {
        for (int i = 0; i < mat_size_; i++) {
            prob_mat_[i] = 0.0f;
        }
    }
} // end of DarknetHelpers namespace


image DarknetImageHolder::CvMatToImage(const cv::Mat &cv_image,
                                       const int target_width,
                                       const int target_height) {

    // Darknet uses its own image type, which is a C struct.
    image tmp_image = make_image(cv_image.cols,
                                 cv_image.rows,
                                 cv_image.channels());
    
    // This code is mostly copied from Darknet's "ipl_into_image"
    // function. The main difference is that this function works
    // with cv::Mat instead of IplImage. IplImage is a legacy
    // OpenCV image type. The OpenCV documentation for cv::Mat
    // says that cv::Mat and IplImage use a compatible data
    // layout.
    int width = tmp_image.w;
    int height = tmp_image.h;
    int channels = cv_image.channels();
    size_t step = cv_image.step[0];

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int channel = 0; channel < channels; channel++) {
                tmp_image.data[channel * width * height + row * width + col]
                        = cv_image.data[row * step + col * channels + channel] / 255.0f;
            }
        }
    }
    rgbgr_image(tmp_image);
    image darknet_image = letterbox_image(tmp_image, target_width, target_height);
    free_image(tmp_image);
    return darknet_image;
}

    DarknetImageHolder::DarknetImageHolder(const bool s) 
            : index(0), stop_flag(s), orig_frame_size(cv::Size(0,0)) {
        darknet_image.w = 0;
        darknet_image.h = 0;
        darknet_image.c = 0;
        darknet_image.data = NULL;
    }

    DarknetImageHolder::DarknetImageHolder(const cv::Mat &cv_image,
                                           const int target_width,
                                           const int target_height)
            : index(0), stop_flag(false) {
        orig_frame_size = cv_image.size();
        darknet_image = CvMatToImage(cv_image, target_width, target_height);
    }

    DarknetImageHolder::DarknetImageHolder(int frame_num,
                                           const cv::Mat &cv_image,
                                           const int target_width,
                                           const int target_height)
            : index(frame_num), stop_flag(false) {
        orig_frame_size = cv_image.size();
        darknet_image = CvMatToImage(cv_image, target_width, target_height);
    }

    DarknetImageHolder::~DarknetImageHolder() {
        if (NULL != darknet_image.data) {
            free_image(darknet_image);
        }
    }


namespace {

    // Darknet functions that accept C style strings, accept a char*
    // instead of a const char*.
    std::unique_ptr<char[]> ToNonConstCStr(const std::string &str) {
        std::unique_ptr<char[]> result(new char[str.size() + 1]);
        size_t length = str.copy(result.get(), str.size());
        result[length] = '\0';
        return result;
    }


    DarknetHelpers::network_ptr_t LoadNetwork(const ModelSettings &model_settings) {
        auto cfg_file = ToNonConstCStr(model_settings.network_config_file);
        auto weights_file = ToNonConstCStr(model_settings.weights_file);

        return { load_network(cfg_file.get(), weights_file.get(), 0),
                    free_network };
    }


    int GetOutputLayerSize(const network &network) {
        layer output_layer = network.layers[network.n - 1];
        return output_layer.w * output_layer.h * output_layer.n;
    }


    int GetNumClasses(const network &network) {
        layer output_layer = network.layers[network.n - 1];
        return output_layer.classes;
    }


    std::vector<std::string> LoadNames(const ModelSettings &model_settings, int expected_name_count) {
        std::ifstream in_file(model_settings.names_file);
        if (!in_file.good()) {
            throw std::runtime_error("Failed to open names file at: " + model_settings.names_file);
        }

        std::vector<std::string> names;
        names.reserve(static_cast<size_t>(expected_name_count));

        std::string line;
        while (std::getline(in_file, line).good()) {
            names.push_back(line);
        }

        if (names.size() != expected_name_count) {
            std::stringstream error;
            error << "Error: The network config file at " << model_settings.network_config_file << " specifies "
                  << expected_name_count << " classes, but the names file at " << model_settings.names_file << " contains "
                  << names.size() << " classes. This is probably because given names file does not correspond to the "
                  << "given network configuration file.";
            throw std::runtime_error(error.str());
        }

        return names;
    }


    cv::Rect BoxToRect(const box &box, const cv::Size &image_size) {
        // box.x and box.y refer to the center of the rectangle,
        // but cv::Rect uses the top left x and y coordinates.
        auto tlX = static_cast<int>(box.x - box.w / 2.0f);
        auto tlY = static_cast<int>(box.y - box.h / 2.0f);
        auto width = static_cast<int>(box.w);
        auto height = static_cast<int>(box.h);

        return cv::Rect(tlX, tlY, width, height) & cv::Rect(cv::Point(0, 0), image_size);
    }


    bool HasWhitelist(const Properties &props) {
        return !DetectionComponentUtils::GetProperty(props, "CLASS_WHITELIST_FILE", std::string())
                .empty();
    }


    class NoOpFilter {
    public:
        NoOpFilter(const std::map<std::string, std::string> &, const std::vector<std::string>&) { }

        bool operator()(const std::string &) { return true; }
    };


    class WhitelistFilter {
    public:

        WhitelistFilter(const Properties& props, const std::vector<std::string>& names) {
            const std::string &whitelist_path = props.at("CLASS_WHITELIST_FILE");
            std::string expanded_file_path;
            std::string error = Utils::expandFileName(whitelist_path, expanded_file_path);
            if (!error.empty()) {
                throw MPFInvalidPropertyException(
                    "CLASS_WHITELIST_FILE",
                    "The value, \"" + whitelist_path + "\", could not be expanded due to: " + error);
            }

            std::ifstream whitelist_file(expanded_file_path);
            if (!whitelist_file.good()) {
                throw MPFDetectionException(
                    MPF_COULD_NOT_OPEN_DATAFILE,
                    "Failed to load class whitelist that was supposed to be located at \""
                    + expanded_file_path + "\".");
            }

            std::unordered_set<std::string> temp_whitelist;
            std::string line;
            while (std::getline(whitelist_file, line)) {
                Utils::trim(line);
                if (!line.empty()) {
                    temp_whitelist.insert(line);
                }
            }

            if (temp_whitelist.empty()) {
                throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "The class whitelist file located at \"" + expanded_file_path + "\" was empty.");
            }

            for (const std::string &name : names) {
                if (temp_whitelist.count(name) > 0) {
                    whitelist_.insert(name);
                }
            }

            if (whitelist_.empty()) {
                throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "None of the class names specified in the whitelist file located at \""
                    + expanded_file_path + "\" were found in the names file.");
            }
        }

        bool operator()(const std::string& class_name) {
            return whitelist_.count(class_name) > 0;
        }
    private:
        std::unordered_set<std::string> whitelist_;
    };

}  // end of anonymous namespace


template<typename ClassFilter>
DarknetImpl<ClassFilter>::DarknetImpl(const std::map<std::string, std::string> &props, const ModelSettings &settings)
        : DarknetInterface(props, settings)
    , network_(LoadNetwork(settings))
    , output_layer_size_(GetOutputLayerSize(*network_))
    , num_classes_(GetNumClasses(*network_))
    , names_(LoadNames(settings, num_classes_))
    , class_filter_(props, names_)
    // Darknet will output of a probability for every possible class regardless of the content of the image.
    // Most of these classes will have a probability of zero or a number very close to zero.
    // If the confidence threshold is zero or smaller it will report every possible classification.
    , confidence_threshold_(DetectionComponentUtils::GetProperty(props, "CONFIDENCE_THRESHOLD", 0.5f))
    , probs_(output_layer_size_, num_classes_)
    , boxes_(new box[output_layer_size_]) {}


template<typename ClassFilter>
void DarknetImpl<ClassFilter>::ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                               std::vector<MPFImageLocation> &locations) {

    std::unordered_map<std::string, MPFImageLocation> type_to_image_loc;

    for (DarknetResult &darknet_result : darknet_results) {
        const cv::Rect &rect = darknet_result.detection_rect;

        for (std::pair<float, std::string> &class_prob : darknet_result.object_type_probs) {
            auto it = type_to_image_loc.find(class_prob.second);
            if (it == type_to_image_loc.cend()) {
                type_to_image_loc.emplace(
                        class_prob.second,
                        MPFImageLocation(rect.x, rect.y, rect.width, rect.height, class_prob.first,
                                         Properties{ {"CLASSIFICATION", class_prob.second} }));
            }
            else {
                TrackingHelpers::CombineImageLocation(rect, class_prob.first, it->second);
            }
        }
    }

    for (auto &image_loc_pair : type_to_image_loc) {
        locations.push_back(std::move(image_loc_pair.second));
    }
}

// This function runs the detection and tracking on a separate thread.
template<typename ClassFilter>
template<typename Tracker>
void DarknetImpl<ClassFilter>::ProcessFrameQueue(Tracker &tracker,
                                                 DarknetQueue &queue) {

    std::vector<DarknetResult>detections;

    do {
        std::unique_ptr<DarknetImageHolder> current_frame;

        try {
            current_frame = queue.pop();
        }
        catch (const std::runtime_error &e) {
            // The pop() function will throw an exception when the
            // queue has been halted.
            break;
        }

        if (current_frame->stop_flag) {
            // The last frame for this job has been processed. No more
            // to do.
            break;
        }

        // Process it
        tracker.ProcessFrameDetections(Detect(*current_frame),
                                        current_frame->index);
    } while (1);
}

template<typename ClassFilter>
void DarknetImpl<ClassFilter>::FinishDetection(std::thread &detection_thread,
                                               DarknetQueue &queue, bool halt) {
    if (halt) queue.halt();
    if (detection_thread.joinable()) {
        detection_thread.join();
    }
}

template<typename ClassFilter>
void DarknetImpl<ClassFilter>::EnqueueStopFrame(DarknetQueue &queue) {
    // Put a stop frame into the queue to tell the consumer it is done.
    std::unique_ptr<DarknetImageHolder> stop_frame(new DarknetImageHolder(true));
    try {
        queue.push(std::move(stop_frame));
    }
    catch (const std::runtime_error &e) {
        return;
    }
}


template<typename ClassFilter>
MPFDetectionError DarknetImpl<ClassFilter>::ReadAndEnqueueFrames(MPFVideoCapture &video_cap, DarknetQueue &queue) {

    try {
        cv::Mat frame;
        int frame_number = -1;

        if (!video_cap.Read(frame)) {
            return MPF::COMPONENT::MPFDetectionError::MPF_COULD_NOT_READ_DATAFILE;
        }

        do {
            frame_number++;

            std::unique_ptr<DarknetImageHolder> current_frame(new DarknetImageHolder(frame_number, frame, network_->w, network_->h));

            try {

                queue.push(std::move(current_frame));
            }
            catch (const std::runtime_error &e) {
                break;
            }

        } while (video_cap.Read(frame));

        EnqueueStopFrame(queue);
    }
    catch (std::exception &e) {
        return MPF::COMPONENT::MPFDetectionError::MPF_COULD_NOT_READ_DATAFILE;
    }
    return MPF_DETECTION_SUCCESS;
}

template<typename ClassFilter>
MPFDetectionError DarknetImpl<ClassFilter>::RunDarknetDetection(const MPFVideoJob &job,
                                                                std::vector<MPFVideoTrack> &tracks) {
    
    if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
        PreprocessorTracker tracker;
        return RunDarknetDetection(job, tracks, tracker);
    }
    else {
        int number_of_classifications = DetectionComponentUtils::GetProperty(
                job.job_properties, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5);
        double rect_min_overlap = DetectionComponentUtils::GetProperty(
                job.job_properties, "MIN_OVERLAP", 0.5);

        DefaultTracker tracker(number_of_classifications, rect_min_overlap);
        return RunDarknetDetection(job, tracks, tracker);
    }
}

template<typename ClassFilter>
template<typename Tracker>
MPFDetectionError DarknetImpl<ClassFilter>::RunDarknetDetection(const MPFVideoJob &job,
                                                                std::vector<MPFVideoTrack> &tracks,
                                                                Tracker &tracker) {

    MPFVideoCapture video_cap(job);
    if (!video_cap.IsOpened()) {
        return MPF_COULD_NOT_OPEN_DATAFILE;
    }

    DarknetQueue queue(DetectionComponentUtils::GetProperty(job.job_properties, "FRAME_QUEUE_CAPACITY", 4));

    std::thread detection_thread = std::thread{ [this, &tracker, &queue]() { DarknetImpl::ProcessFrameQueue(tracker, queue); } };

    MPFDetectionError rc = ReadAndEnqueueFrames(video_cap, queue);
    if (rc != MPF_DETECTION_SUCCESS) {
        // Set the halt parameter to true so that the detection thread
        // won't be waiting forever for more frames to be sent.
        FinishDetection(detection_thread, queue, true);
    }
    else {
        // Wait for the thread to finish, setting the halt parameter
        // to false so that it will execute all the way to the stop
        // frame before returning.
        FinishDetection(detection_thread, queue, false);
        tracks = tracker.GetTracks();

        for (MPFVideoTrack &track : tracks) {
            video_cap.ReverseTransform(track);
        }
    }
    return rc;
}


template<typename ClassFilter>
std::vector<DarknetResult> DarknetImpl<ClassFilter>::Detect(const DarknetImageHolder &image_holder) {
    probs_.Clear();

    // There is no documentation explaining what hier_thresh and nms do,
    // so we are just using the default values from the Darknet library.
    float hier_thresh = 0.5;
    float nms = 0.3;
    network *net = network_.get();
    image im = image_holder.darknet_image;
    set_batch_network(net, 1);
    network_predict(net, im.data);

    layer l = net->layers[net->n-1];
    if(l.type == REGION){
        int image_width = image_holder.orig_frame_size.width;
        int image_height = image_holder.orig_frame_size.height;
        get_region_boxes(l, image_width, image_height, net->w, net->h,
                         confidence_threshold_, probs_.Get(),
                         boxes_.get(), 0, 0, 0, hier_thresh, 0);
        do_nms_sort(boxes_.get(), probs_.Get(), l.w*l.h*l.n, l.classes, nms);
    }

    std::vector<DarknetResult> detections;

    for (int box_idx = 0; box_idx < output_layer_size_; box_idx++) {
        DarknetResult detection { BoxToRect(boxes_[box_idx], image_holder.orig_frame_size) };

        for (int name_idx = 0; name_idx < num_classes_; name_idx++) {
            float prob = probs_[box_idx][name_idx];
            if (prob >= confidence_threshold_ && class_filter_(names_[name_idx])) {
                detection.object_type_probs.emplace_back(prob, names_[name_idx]);
            }
        }

        if (!detection.object_type_probs.empty()) {
            detections.push_back(std::move(detection));
        }
    }

    return detections;
}

template<typename ClassFilter>
MPFDetectionError DarknetImpl<ClassFilter>::RunDarknetDetection(const MPFImageJob &job,
                                                                std::vector<MPFImageLocation> &locations) {

    MPFImageReader image_reader(job);
    DarknetImageHolder image(image_reader.GetImage(), network_->w, network_->h);
    std::vector<DarknetResult> results = Detect(image);
    if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
        ConvertResultsUsingPreprocessor(results, locations);
    }
    else {
        int number_of_classifications
                = std::max(1, DetectionComponentUtils::GetProperty(job.job_properties,
                                                                   "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5));
        for (auto& result : results) {
            locations.push_back(
                TrackingHelpers::CreateImageLocation(number_of_classifications, result));
        }
    }

    for (auto &location : locations) {
        image_reader.ReverseTransform(location);
    }

    return MPF_DETECTION_SUCCESS;
}


void configure_cuda_device(const Properties &job_props) {
#ifdef GPU
    int cuda_device_id = DetectionComponentUtils::GetProperty(job_props, "CUDA_DEVICE_ID", -1);
    if (cuda_device_id < 0) {
        throw MPFDetectionException(
                MPF_GPU_ERROR, "CUDA version of darknet library loaded, but the CUDA_DEVICE_ID was not set.");
    }

    gpu_index = cuda_device_id;
    cudaError_t rc = cudaSetDevice(cuda_device_id);
    if (rc != cudaSuccess) {
        throw MPFDetectionException(
                MPF_GPU_ERROR,
                "Failed to set CUDA device to device number " + std::to_string(cuda_device_id)
                         + " due to: " + cudaGetErrorString(rc));
    }
    // Through testing we have determined that the following function
    // must be called after cudaSetDevice() in order for it to take
    // effect on the device just selected. This seems contrary to what
    // is implied by the documentation, specifically regarding calling
    // it before the runtime and driver have been initialized.
    // In addition, our testing has been unable to find a circumstance
    // where this function fails, also despite what the documentation
    // says. For this reason, we treat failure of this function as a
    // fatal error, since it should not fail under normal operation.
    rc = cudaSetDeviceFlags(cudaDeviceBlockingSync);
    if (rc != cudaSuccess) {
        throw MPFDetectionException(MPF_GPU_ERROR, "Could not set CUDA device " + std::to_string(cuda_device_id) + " to use blocking synchronization: " + cudaGetErrorString(rc));
    }
#endif
}


extern "C" {
    DarknetInterface* darknet_impl_creator(const std::map<std::string, std::string> *props,
                                           const ModelSettings *settings) {
        configure_cuda_device(*props);
        if (HasWhitelist(*props)) {
            return new DarknetImpl<WhitelistFilter>(*props, *settings);
        }
        else {
            return new DarknetImpl<NoOpFilter>(*props, *settings);
        }
    }

    void darknet_impl_deleter(DarknetInterface *impl) {
        delete impl;
    }
}
