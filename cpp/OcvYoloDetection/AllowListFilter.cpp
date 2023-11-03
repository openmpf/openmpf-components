/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

#include "AllowListFilter.h"

#include <fstream>

#include <Utils.h>
#include <MPFInvalidPropertyException.h>
#include <MPFDetectionObjects.h>


using namespace MPF::COMPONENT;

namespace {

    std::unordered_set<std::string> LoadAllowlist(
            const std::string &allowListPath, const std::vector<std::string> &names) {
        std::string expandedFilePath;
        std::string error = Utils::expandFileName(allowListPath, expandedFilePath);
        if (!error.empty()) {
            throw MPFInvalidPropertyException(
                    "CLASS_ALLOW_LIST_FILE",
                    "The value, \"" + allowListPath + "\", could not be expanded due to: "
                    + error);
        }

        std::ifstream allowListFile(expandedFilePath);
        if (!allowListFile.good()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_OPEN_DATAFILE,
                    "Failed to load class allow list that was supposed to be located at \""
                    + expandedFilePath + "\".");
        }

        std::unordered_set<std::string> tempAllowList;
        std::string line;
        while (std::getline(allowListFile, line)) {
            Utils::trim(line);
            if (!line.empty()) {
                tempAllowList.insert(std::move(line));
                line = "";
            }
        }

        if (tempAllowList.empty()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "The class allow list file located at \"" + expandedFilePath + "\" was empty.");
        }

        std::unordered_set<std::string> allowList;
        for (const std::string &name: names) {
            if (tempAllowList.count(name) > 0) {
                allowList.insert(name);
            }
        }

        if (allowList.empty()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "None of the class names specified in the allow list file located at \""
                    + expandedFilePath + "\" were found in the names file.");
        }
        return allowList;
    }
}


AllowListFilter::AllowListFilter(const std::string &allowListPath,
                                 const std::vector<std::string> &names)
        : allowList_(LoadAllowlist(allowListPath, names)) {
}


bool AllowListFilter::operator()(const std::string &className) const {
    return allowList_.count(className) > 0;
}
