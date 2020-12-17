/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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

/******************************************************************************
 * File:        tessdata_model_updater.cpp                                *
 * Description: Updates tesseract traineddata files from user provided        *
 *              wordlist and model files.                                     *
 *                                                                            *
 *              Based on the following Tesseract training tools, with         *
 *              additional processing for custom wordlists and model files.   *
 *                                                                            *
 * Reference:   tesseract/src/training/combine_tessdata.cpp                   *
 * Description: Creates a unified traineddata file from several               *
 *              data files produced by the training process.                  *
 * Author:      Daria Antonova                                                *
 *                                                                            *
 * Reference:   tesseract/src/training/dawg2wordlist.cpp                      *
 * Description: Program to create a word list from a DAWG and unicharset.     *
 * Author:      David Eger                                                    *
 *                                                                            *
 * Reference:   tesseract/src/training/wordlist2dawg.cpp                      *
 * Description: Program to generate a DAWG from a word list file.             *
 * Author:      Thomas Kielbus                                                *
 ******************************************************************************/

#ifndef TESSDATA_MODEL_UPDATER_H
#define TESSDATA_MODEL_UPDATER_H

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <set>
#include <vector>
#include <utility>
#include <unordered_map>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/algorithm/string.hpp>

#include <tesseract/baseapi.h>

namespace MPF_Model_Updater {

    void extractLangModel(std::string path_to_model, std::string output_prefix);

    void updateLanguageModel(const std::string &path_to_model, char** updated_files, const int &updated_num_files);

    tesseract::Dawg* LoadSquishedDawg(const UNICHARSET &unicharset, const char *filename);

    void convertWordListToDawg(const char *unicharset_file, const char *wordlist_file, const char *dawg_file);

    void convertDawgToWordList(const char *unicharset_file, const char *dawg_file, const char *wordlist_file);

    void addToWordList(const char *wordlist_file, std::set<std::string> &output_wordset);

    void combineWordLists(const char *wordlist_file1, const char *wordlist_file2, const char *output_file);

    void addWordListToDawg(const char *unicharset_file,
                           const char *wordlist_file,
                           const char *dawg_file);

    void copyWordListOverDawg(const char *unicharset_file,
                              const char *wordlist_file,
                              const char *dawg_file);

    void checkModels(const char *model_dir, std::set<std::string> &model_set, bool &is_empty);

    bool checkDictDir(const char *dict_dir,
                      std::set<std::string> &original_models,
                      std::set<std::string> &updated_models,
                      std::unordered_map<std::string, std::pair<std::vector<std::string>,
                      std::vector<std::string>>> &lang_dict_map);

    std::set<std::string> updateLanguageFiles(const char *model_dir,
                                              const char *dict_dir,
                                              const char *updated_model_dir,
                                              bool force_update=true,
                                              bool replace_dawgs=false);

    class WordOutputter {
        public:
            WordOutputter(FILE *file);
            void output_word(const char *word);
        private:
            FILE *file_;
    };

    class TempRandomDirectory {
        public:
            boost::filesystem::path path;
            explicit TempRandomDirectory(const std::string &directory);

            ~TempRandomDirectory();
            TempRandomDirectory(const TempRandomDirectory&) = delete;
            TempRandomDirectory& operator=(const TempRandomDirectory&) = delete;
    };

    class TempRandomFile {
        public:
            boost::filesystem::path path;
            explicit TempRandomFile(const std::string &filepath);

            ~TempRandomFile();
            TempRandomFile(const TempRandomFile&) = delete;
            TempRandomFile& operator=(const TempRandomFile&) = delete;
    };
}



#endif //TESSDATA_MODEL_UPDATER_H
