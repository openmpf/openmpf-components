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


#ifndef OPENMPF_COMPONENTS_DARKNETIMPL_H
#define OPENMPF_COMPONENTS_DARKNETIMPL_H

#include <atomic>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>

#include <MPFVideoCapture.h>
#include <BlockingQueue.h>

#include "../include/DarknetInterface.h"

#include "darknet.h"


namespace DarknetHelpers {
    using network_ptr_t = std::unique_ptr<network, decltype(&free_network)>;


    // Darknet's network_detect function has a float** parameter that is used to store the probabilities.
    // The float** is used as a two dimensional array. The first level index is the id of a box,
    // and the second level index is the id of a classification.
    class ProbHolder {
    public:
        ProbHolder(int output_layer_size, int num_classes);

        float** Get();

        float* operator[](size_t box_idx);

        void Clear();

    private:
        int mat_size_;

        std::unique_ptr<float[]> prob_mat_;

        std::unique_ptr<float*[]> prob_row_ptrs_;
    };
}


struct DarknetImageHolder {
    int index;
    bool stop_flag;
    cv::Size orig_frame_size;
    image darknet_image;

    DarknetImageHolder() = delete;
    explicit DarknetImageHolder(const bool s);
    DarknetImageHolder(const cv::Mat &cv_image,
                       const int target_width, const int target_height);
    DarknetImageHolder(int frame_num, const cv::Mat &cv_image,
                       const int target_width, const int target_height);

    ~DarknetImageHolder();

    DarknetImageHolder(const DarknetImageHolder& im) = delete;
    DarknetImageHolder& operator=(const DarknetImageHolder& im) = delete;
    DarknetImageHolder(DarknetImageHolder&& im) = delete;
    DarknetImageHolder& operator=(DarknetImageHolder&& im) = delete;

    static image CvMatToImage(const cv::Mat &cv_image, const int w, const int h);
};


template<typename ClassFilter>
class DarknetImpl : public DarknetInterface {

public:
    DarknetImpl(const MPF::COMPONENT::Properties &props, const ModelSettings &settings);

    MPF::COMPONENT::MPFDetectionError RunDarknetDetection(const MPF::COMPONENT::MPFVideoJob &job,
                                                          std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    MPF::COMPONENT::MPFDetectionError RunDarknetDetection(const MPF::COMPONENT::MPFImageJob &job,
                                                          std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;

private:
    DarknetHelpers::network_ptr_t network_;
    int output_layer_size_;
    int num_classes_;
    std::vector<std::string> names_;
    ClassFilter class_filter_;

    float confidence_threshold_;

    DarknetHelpers::ProbHolder probs_;

    std::unique_ptr<box[]> boxes_;

    log4cxx::LoggerPtr logger_;

    template <typename Tracker>
    MPF::COMPONENT::MPFDetectionError RunDarknetDetection(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
            Tracker &tracker);

    std::vector<DarknetResult> Detect(const DarknetImageHolder &image);

    static void ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                std::vector<MPF::COMPONENT::MPFImageLocation> &locations);

    using DarknetQueue = MPF::COMPONENT::BlockingQueue<std::unique_ptr<DarknetImageHolder>>;

    // ProcessFrameQueue is the function that will be executed by the
    // thread spawned in GetDetections for video jobs.
    template <typename Tracker>
    void ProcessFrameQueue(Tracker &tracker,  DarknetQueue &queue);

   // FinishDetection is used to stop the detection thread. Set the
   // halt parameter to true when the processing must be terminated
   // due to an exception or other error, and the thread will
   // terminate without emptying the queue. Otherwise, the thread will
   // empty the queue before being joined in this function.
    void FinishDetection(std::thread &thread, DarknetQueue &queue, bool halt);

    // This function reads frames from the video and puts them into
    // the queue.
    MPF::COMPONENT::MPFDetectionError ReadAndEnqueueFrames(MPF::COMPONENT::MPFVideoCapture &video_cap, DarknetQueue &queue);

    // This function puts a stop frame into the queue to tell the
    // detecton thread when there are no more frames to be processed.
    void EnqueueStopFrame(DarknetQueue &queue);
};


#endif //OPENMPF_COMPONENTS_DARKNETIMPL_H
