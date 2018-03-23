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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "ModelsIniParser.h"


IniHelper::IniHelper(const std::string &file_path, const std::string &model_name)
        : model_name_(model_name)
{
    using namespace boost::property_tree;
    iptree all_models_ini;
    try {
        ini_parser::read_ini(file_path, all_models_ini);
    }
    catch (const ini_parser::ini_parser_error &ex) {
        throw ModelsIniException("Failed to open \"" + file_path + "\" due to: " + ex.what());
    }


    const auto &model_iter = all_models_ini.find(model_name);
    if (model_iter == all_models_ini.not_found() || model_iter->second.empty()) {
        throw ModelsIniException(
                "Failed to load model \"" + model_name
                + "\" because the models.ini file did not contain a non-empty section named ["
                + model_name + "].");
    }

    for (const auto &sub_tree : model_iter->second) {
        model_ini_fields_.emplace(sub_tree.first, sub_tree.second.data());
    }
}

std::string IniHelper::GetValue(const std::string &key) const {
    try {
        return model_ini_fields_.at(key);
    }
    catch (std::out_of_range &ex) {
        throw ModelsIniException(
                "Unable load the \"" + model_name_ + "\" model because the \"" + key
                + "\" key was not present in the [" + model_name_ + "] section.");
    }
}