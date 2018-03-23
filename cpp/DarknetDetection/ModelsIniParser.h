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


#ifndef OPENMPF_COMPONENTS_MODELSINIPARSER_H
#define OPENMPF_COMPONENTS_MODELSINIPARSER_H

#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <memory>
#include <unordered_map>


class ModelsIniException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};


// In order to avoid including the boost headers in this header,
// everything that uses boost is in this class's implementation.
// This makes it so the boost related code gets compiled as part of the utils library instead of as part of
// the library that includes this.
// Something about the boost property tree headers, probably their complexity,
// prevents clang-tidy from working correctly. When the boost property tree headers were included in this header,
// the clang-tidy issue spread to all files that included this file.
class IniHelper {
public:
    IniHelper(const std::string &file_path, const std::string &model_name);
    std::string GetValue(const std::string &key) const;

private:
    std::string model_name_;
    std::unordered_map<std::string, std::string> model_ini_fields_;
};


template <typename TModelInfo>
class ModelsIniParser {

public:
    typedef std::string (TModelInfo::*class_field_t);


    ModelsIniParser& Init(const std::string &plugin_models_dir) {
        plugin_models_dir_ = plugin_models_dir;
        return *this;
    }

    ModelsIniParser& RegisterField(const std::string &key_name, class_field_t field) {
        if (key_name.empty()) {
            throw ModelsIniException("\"key_name\" must not be empty.");
        }
        fields_.emplace_back(key_name, field);
        return *this;
    }


    TModelInfo ParseIni(const std::string &model_name, const std::string &common_models_dir) const {
        std::string models_ini_path = GetFullPath("models.ini", common_models_dir);
        IniHelper helper(models_ini_path, model_name);

        TModelInfo model_info;
        for (const std::pair<std::string, class_field_t>& field_info : fields_) {
            std::string file_name = helper.GetValue(field_info.first);
            model_info.*field_info.second = GetFullPath(file_name, common_models_dir);
        }
        return model_info;
    }



private:
    std::string plugin_models_dir_;

    std::vector<std::pair<std::string, class_field_t>> fields_;


    std::string GetFullPath(const std::string &file_name, const std::string &common_models_dir) const {

        std::vector<std::string> possible_locations;
        if (file_name.front() == '/') {
            possible_locations.push_back(file_name);
        }
        else {
            possible_locations.push_back(common_models_dir + '/' + file_name);
            possible_locations.push_back(plugin_models_dir_ + '/' + file_name);
        }


        for (const auto& possible_location : possible_locations) {
            if (std::ifstream(possible_location).good()) {
                return possible_location;
            }
        }

        std::string error_msg = "Failed to load model because a required file was not present. ";
        if (file_name.front() == '/') {
            error_msg += "Expected a file at \"" + file_name + "\" to exist.";
        }
        else {
            error_msg += "Expected a file to exist at either \"" + possible_locations.at(0)
                         + "\" or \"" + possible_locations.at(1) + "\".";
        }

        throw ModelsIniException(error_msg);
    }
};


#endif //OPENMPF_COMPONENTS_MODELSINIPARSER_H
