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

#include "WhitelistFilter.h"

#include <fstream>

#include <Utils.h>
#include <MPFInvalidPropertyException.h>
#include <MPFDetectionObjects.h>


using namespace MPF::COMPONENT;

namespace {

    std::unordered_set<std::string> LoadWhitelist(
            const std::string &whiteListPath, const std::vector<std::string> &names) {
        std::string expandedFilePath;
        std::string error = Utils::expandFileName(whiteListPath, expandedFilePath);
        if (!error.empty()) {
            throw MPFInvalidPropertyException(
                    "CLASS_WHITELIST_FILE",
                    "The value, \"" + whiteListPath + "\", could not be expanded due to: "
                    + error);
        }

        std::ifstream whitelistFile(expandedFilePath);
        if (!whitelistFile.good()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_OPEN_DATAFILE,
                    "Failed to load class whitelist that was supposed to be located at \""
                    + expandedFilePath + "\".");
        }

        std::unordered_set<std::string> tempWhitelist;
        std::string line;
        while (std::getline(whitelistFile, line)) {
            Utils::trim(line);
            if (!line.empty()) {
                tempWhitelist.insert(std::move(line));
                line = "";
            }
        }

        if (tempWhitelist.empty()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "The class whitelist file located at \"" + expandedFilePath + "\" was empty.");
        }

        std::unordered_set<std::string> whitelist;
        for (const std::string &name: names) {
            if (tempWhitelist.count(name) > 0) {
                whitelist.insert(name);
            }
        }

        if (whitelist.empty()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "None of the class names specified in the whitelist file located at \""
                    + expandedFilePath + "\" were found in the names file.");
        }
        return whitelist;
    }
}


WhitelistFilter::WhitelistFilter(const std::string &whiteListPath,
                                 const std::vector<std::string> &names)
        : whitelist_(LoadWhitelist(whiteListPath, names)) {
}


bool WhitelistFilter::operator()(const std::string &className) {
    return whitelist_.count(className) > 0;
}
