/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2017 The MITRE Corporation                                       *
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
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <QCoreApplication>
#include <log4cxx/simplelayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include "detectionComponentUtils.h"
#include "LicensePlateTextDetection.h"

using namespace MPF;
using namespace COMPONENT;

inline bool sort_by_start_frame(
        const MPFVideoTrack& track1, const MPFVideoTrack& track2);

int main(int argc, char* argv[]) {
    QCoreApplication *this_app = new QCoreApplication(argc, argv);
    std::string app_dir = (this_app->applicationDirPath()).toStdString();
    delete this_app;

    std::map<std::string, std::string> algorithm_properties;
    MPFDetectionDataType media_type = IMAGE;
    if (argc > 2) {
        media_type = VIDEO;
    }
    std::string uri(argv[1]);

    // set up logger
    log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("logger");
    log4cxx::SimpleLayoutPtr layout = new log4cxx::SimpleLayout();
    log4cxx::ConsoleAppenderPtr console_appender =
            new log4cxx::ConsoleAppender(layout);
    logger->addAppender(console_appender);
    logger->setLevel(log4cxx::Level::getDebug());

    // instantiate text detection component
    LicensePlateTextDetection te;
    te.SetRunDirectory(app_dir + "/plugin");
    if (te.Init()) {

        switch(media_type) {

            case IMAGE:  //case 1
            {
                std::vector<MPFImageLocation> detections;
                MPFImageJob job("Testing", uri, algorithm_properties, { });
                te.GetDetections(job, detections);
                for (int i = 0; i < detections.size(); i++) {
                    LOG4CXX_INFO(logger, "detection number "
                            << i
                            << " Text is "
                            << DetectionComponentUtils::GetProperty<std::string>(detections[i].detection_properties, "TEXT", ""));
                }
                break;
            }

            case VIDEO:  //case 0
            {
                int start_frame = atoi(argv[2]);
                int stop_frame = atoi(argv[3]);
                int FRAME_INTERVAL = 1;
                if (argv[4]) {
                    FRAME_INTERVAL = atoi(argv[4]);
                }
                std::ostringstream stringStream;
                stringStream << FRAME_INTERVAL;
                algorithm_properties.insert(std::pair<std::string, std::string>(
                        "FRAME_INTERVAL", stringStream.str()));
                std::vector<MPFVideoTrack> tracks;
                MPFVideoJob job("Testing", uri, start_frame, stop_frame, algorithm_properties, { });
                te.GetDetections(job, tracks);
                std::sort(tracks.begin(), tracks.end(), sort_by_start_frame);
                LOG4CXX_INFO(logger,
                             "number of tracks is " << tracks.size());
                for (int i = 0; i < tracks.size(); i++) {
                    LOG4CXX_INFO(logger, "start frame = " << tracks[i].start_frame
                                                          << ", stop frame = " << tracks[i].stop_frame
                                                          << ", detection vector size is " << tracks[i].frame_locations.size()
                                                          << ", text is "
                                                          << DetectionComponentUtils::GetProperty<std::string>(tracks[i].detection_properties, "TEXT", ""));
                    for (std::map<int,MPFImageLocation>::iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it) {
                        LOG4CXX_DEBUG(logger, it->first << ","
                                                        << it->second.x_left_upper << ","
                                                        << it->second.y_left_upper << ","
                                                        << it->second.width << ","
                                                        << it->second.height);
                    }
                }
                break;
            }

            default:
            {
                LOG4CXX_ERROR(logger, "Error - invalid detection type");
                break;
            }
        }

    } else {
        LOG4CXX_ERROR(logger, "Error - could not initialize text detection component");
    }

    te.Close();
    return 1;
}

bool sort_by_start_frame(const MPFVideoTrack& track1, const MPFVideoTrack& track2) {
    return (track1.start_frame < track2.start_frame);
}
