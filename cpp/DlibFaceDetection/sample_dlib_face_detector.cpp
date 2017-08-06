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

#include <dlfcn.h>
#include <string>
#include <vector>
#include <log4cxx/simplelayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include "detectionComponentUtils.h"

using namespace MPF::COMPONENT;

int main(int argc, char* argv[]) {
    log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("logger");
    log4cxx::SimpleLayoutPtr layout = new log4cxx::SimpleLayout();
    log4cxx::ConsoleAppenderPtr console_appender =
            new log4cxx::ConsoleAppender(layout);
    logger->addAppender(console_appender);
    logger->setLevel(log4cxx::Level::getDebug());

    //TODO: test IMAGE and VIDEO

    std::map<std::string, std::string> algorithm_properties;
    MPFDetectionDataType media_type = IMAGE;
    std::string uri(argv[2]);
    LOG4CXX_INFO(logger, "uri is " << uri);

    void *component_handle = dlopen(argv[1], RTLD_NOW);
    if (NULL == component_handle) {
        LOG4CXX_ERROR(logger, "Failed to open component library " << argv[1] << ": " << dlerror());
        return -1;
    }

    typedef MPFComponent* create_t();
    typedef void destroy_t(MPFComponent*);

    create_t* comp_create = (create_t*)dlsym(component_handle, "component_creator");
    if (NULL == comp_create) {
        LOG4CXX_ERROR(logger, "dlsym failed for component_creator: " << dlerror());
        dlclose(component_handle);
        return -1;
    }

    destroy_t* comp_destroy = (destroy_t*)dlsym(component_handle, "component_deleter");
    if (NULL == comp_destroy) {
        LOG4CXX_ERROR(logger,"dlsym failed for component_deleter: " << dlerror());
        dlclose(component_handle);
        return -1;
    }

    MPFComponent* component_engine = comp_create();
    if (NULL == component_engine) {
        LOG4CXX_ERROR(logger, "Component_engine creation failed: ");
        dlclose(component_handle);
        return -1;
    }

    MPFDetectionComponent* mpf_detection_component =
            dynamic_cast<MPFDetectionComponent*>(component_engine);
    if (mpf_detection_component == NULL) {
        LOG4CXX_ERROR(logger, "Cannot identify component object class");
        return -1;
    }

    component_engine->SetRunDirectory("plugin");
    if (!component_engine->Init()) {
        LOG4CXX_ERROR(logger, "Component initialization failed, exiting.");
        return -1;
    }

    std::vector<MPFImageLocation> detections;
    MPFImageJob job("Testing", uri, algorithm_properties, { });
    mpf_detection_component->GetDetections(job, detections);

    int i = 0;
    for (auto loc : detections) {
        LOG4CXX_INFO(logger, "detection number "
                << i++
                << ": location: ["
                << loc.x_left_upper
                << ", "
                << loc.y_left_upper
                << ", "
                << loc.width
                << ", "
                << loc.height
                << "] confidence is "
                << detections[i].confidence);
    }
    mpf_detection_component->Close();
    return 1;
}

