/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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
 * File:        tessdata_model_updater.cpp                                    *
 * Description: Updates tesseract traineddata files from user provided        *
 *              word lists and model files.                                   *
 *                                                                            *
 *              Written based on the following Tesseract training tools, with *
 *              additional processing for custom word lists and model files.  *
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

#include "tessdata_model_updater.h"
#include "model_updater_tesseract_src/commontraining.h"
#include "model_updater_tesseract_src/lstmrecognizer.h"
#include "model_updater_tesseract_src/classify.h"
#include "model_updater_tesseract_src/tessdatamanager.h"

using namespace MPF_Model_Updater;

/**
 * Helper class for outputting tessdata word lists.
 * @param file - Output path for word list.
 */
WordOutputter::WordOutputter(FILE *file) : file_(file) {}

/**
 * Writes output to word list.
 * @param word - output to word list.
 */
void WordOutputter::output_word(const char *word) {
    fprintf(file_, "%s\n", word);
}

/**
 * Utility class for creating temporary random subdirectories.
 * @param directory - Base path for random temporary subdirectory.
 */
TempRandomDirectory::TempRandomDirectory(const std::string &directory, const std::string &prefix = "/tmp-%"){
    path = boost::filesystem::path(directory)/boost::filesystem::path(prefix + "%%%%%%%%%%%%%%%%%%%%%%%%%");
    path = boost::filesystem::unique_path(path);
    boost::filesystem::create_directories(path);
}

TempRandomDirectory::~TempRandomDirectory() {
    boost::filesystem::remove_all(path);
}

/**
 * Utility class for creating temporary random file names.
 * Any file generated with the given name will be removed once the class object is deleted.
 * @param filepath - Base filepath will receive a temporary ID.
 */
TempRandomFile::TempRandomFile(const std::string &filepath){
    path = boost::filesystem::path(filepath + "_tmp_%%%%%%%%%%%%%%%%%%%%%%%%%%");
    path = boost::filesystem::unique_path(path);
}

TempRandomFile::~TempRandomFile() {
    boost::filesystem::remove_all(path);
}

/**
 * Extracts specified model to output location.
 * @param path_to_model - Specified traineddata model path.
 * @param output_prefix - Output path + prefix of unpacked model files (ex. 'out_path/lang').
 */
void MPF_Model_Updater::extractLangModel(std::string path_to_model, std::string output_prefix) {
    tesseract::CheckSharedLibraryVersion();

    printf("Extracting tessdata components from %s\n", path_to_model.c_str());
    tesseract::TessdataManager tm;

    // Initialize TessdataManager with the data in the given traineddata file.
    if (!tm.Init(path_to_model.c_str())) {
       tprintf("Failed to read %s\n", path_to_model.c_str());
        return;
    }

    // Extract all components.
    for (int i = 0; i < tesseract::TESSDATA_NUM_ENTRIES; ++i) {
        std::string filename = output_prefix;
        char last = filename.back();
        if (last != '.')
            filename += '.';
        filename += tesseract::kTessdataFileSuffixes[i];
        errno = 0;
        if (tm.ExtractToFile(filename.c_str())) {
            printf("Wrote %s\n", filename.c_str());
        } else if (errno != 0) {
            printf("Error, could not extract %s: %s\n", filename.c_str(), strerror(errno));
            return;
        }
    }
    tm.Directory();
}

/**
 * Updates model with given model files. Given files must match model files stored internally within the provided
 * tesseract traineddata file.
 * @param path_to_model - Path of tesseract traineddata file.
 * @param updated_files - List of model files to add to model.
 * @param updated_num_files - Number of model files to add to model.
 */
void MPF_Model_Updater::updateLanguageModel(const std::string &path_to_model,
                                            char** updated_files,
                                            const int &updated_num_files) {
    tesseract::CheckSharedLibraryVersion();
    tesseract::TessdataManager tm;

    std::string traineddata_filename = path_to_model;

    // Initialize TessdataManager with the data in the given traineddata file.
    tm.Init(traineddata_filename.c_str());

    // Write the updated traineddata file.
    tm.OverwriteComponents(traineddata_filename.c_str(), updated_files, updated_num_files);
    tm.Directory();
}

/**
 * Loads DAWG model file using given model unicharset file.
 * Note: The correct unicharset file must be provided or results will be gibberish.
 *
 * @param unicharset - Unicharset file contained within target traineddata model file.
 * @param filename - Name of DAWG file.
 * @return
 */
tesseract::Dawg* MPF_Model_Updater::LoadSquishedDawg(const UNICHARSET &unicharset,
                                                     const char *filename) {
    const int kDictDebugLevel = 1;
    tesseract::TFile dawg_file;
    if (!dawg_file.Open(filename, nullptr)) {
        tprintf("Could not open %s for reading.\n", filename);
        return nullptr;
    }
    tprintf("Loading word list from %s\n", filename);
    tesseract::SquishedDawg *retval = new tesseract::SquishedDawg(
            tesseract::DAWG_TYPE_WORD, "eng", SYSTEM_DAWG_PERM, kDictDebugLevel);
    if (!retval->Load(&dawg_file)) {
        tprintf("Could not read %s\n", filename);
        delete retval;
        return nullptr;
    }
    tprintf("Word list loaded.\n");
    return retval;
}

/**
 * Converts DAWG model file to text format.
 * Note: The correct unicharset file must be provided or results will be gibberish.
 *
 * @param unicharset_file - Unicharset file contained within target traineddata model file.
 * @param dawg_file  - Path of DAWG file.
 * @param wordlist_file - Path to word list text output file.
 */
void MPF_Model_Updater::convertDawgToWordList(const char *unicharset_file,
                                              const char *dawg_file,
                                              const char *wordlist_file) {

    tesseract::CheckSharedLibraryVersion();

    UNICHARSET unicharset;
    if (!unicharset.load_from_file(unicharset_file)) {
        tprintf("Error loading unicharset from %s.\n", unicharset_file);
        return;
    }
    tesseract::Dawg *dict = LoadSquishedDawg(unicharset, dawg_file);
    if (dict == nullptr) {
        tprintf("Error loading dictionary from %s.\n", dawg_file);
        return;
    }

    FILE *out = fopen(wordlist_file, "wb");
    if (out == nullptr) {
        tprintf("Could not open %s for writing.\n", wordlist_file);
        delete dict;
        return;
    }
    WordOutputter outputter(out);
    TessCallback1<const char *> *print_word_cb =
            NewPermanentTessCallback(&outputter, &WordOutputter::output_word);
    dict->iterate_words(unicharset, print_word_cb);
    delete print_word_cb;
    fclose(out);
    delete dict;
    return;
}

/**
 * Converts given word list .txt file back to DAWG format.
 * Note: The correct unicharset file must be provided or results will be gibberish.
 *
 * @param unicharset_file - Unicharset file contained within target traineddata model file.
 * @param wordlist_file - Path to given word list text file.
 * @param dawg_file  - Path of DAWG output file.
 */
void MPF_Model_Updater::convertWordListToDawg(const char *unicharset_file,
                                              const char *wordlist_file,
                                              const char *dawg_file) {

    tesseract::CheckSharedLibraryVersion();
    tesseract::Classify *classify = new tesseract::Classify();
    tprintf("Loading unicharset from '%s'\n", unicharset_file);
    if (!classify->getDict().getUnicharset().load_from_file(unicharset_file)) {
        tprintf("Failed to load unicharset from '%s'\n", unicharset_file);
        delete classify;
        return;
    }
    const UNICHARSET &unicharset = classify->getDict().getUnicharset();
    tesseract::Trie::RTLReversePolicy reverse_policy =
            tesseract::Trie::RRP_DO_NO_REVERSE;

    tesseract::Trie trie(
            // the first 3 arguments are not used in this case
            tesseract::DAWG_TYPE_WORD, "", SYSTEM_DAWG_PERM,
            unicharset.size(), classify->getDict().dawg_debug_level);
    tprintf("Reading word list from '%s'\n", wordlist_file);
    if (!trie.read_and_add_word_list(wordlist_file, unicharset,
                                     reverse_policy)) {
        tprintf("Failed to add word list from '%s'\n", wordlist_file);
        exit(1);
    }
    tprintf("Reducing Trie to SquishedDawg\n");
    tesseract::SquishedDawg *dawg = trie.trie_to_dawg();
    if (dawg != nullptr && dawg->NumEdges() > 0) {
        tprintf("Writing squished DAWG to '%s'\n", dawg_file);
        dawg->write_squished_dawg(dawg_file);
    } else {
        tprintf("DAWG is empty, skip producing the output file\n");
    }
    delete dawg;
    delete classify;
}

/**
 * Helper function for loading word lists in txt format.
 * Trims and ignores blank entries.
 *
 * @param wordlist_file - Path of word list.
 * @param output_wordset - Output word set.
 */
void MPF_Model_Updater::loadWordList(const char *wordlist_file,
                                      std::set<std::string> &output_wordset) {
    std::ifstream in(wordlist_file);
    std::string str;
    while (std::getline(in, str)) {
        boost::trim(str);
        if (str.size() > 0) {
           output_wordset.insert(str);
        }
    }
    in.close();
}


/**
 * Combines two text formatted word lists together.
 *
 * @param wordlist_file1 - First word list file path.
 * @param wordlist_file2 - Second word list file path.
 * @param output_file - Output path of combined word list.
 */
void MPF_Model_Updater::combineWordLists(const char *wordlist_file1,
                                         const char *wordlist_file2,
                                         const char *output_file) {

    std::set <std::string> output_wordset;
    loadWordList(wordlist_file1, output_wordset);
    loadWordList(wordlist_file2, output_wordset);
    std::set<std::string>::iterator it = output_wordset.begin();

    std::ofstream outfile;
    outfile.open(output_file);
    while(it != output_wordset.end()) {
        outfile << *it << "\n" ;
        it++;
    }
    outfile.close();
}

/**
 * Adds a text formatted word list file to a given DAWG file.
 * Note: The correct unicharset file must be provided or results will be gibberish.
 *
 * @param unicharset_file - Unicharset file contained within target traineddata model file.
 * @param wordlist_file - Path to given word list text file.
 * @param dawg_file  - Path to existing DAWG file.
 */
void MPF_Model_Updater::addWordListToDawg(const char *unicharset_file,
                                          const char *wordlist_file,
                                          const char *dawg_file) {


    TempRandomFile tmp_wordlist(std::string(dawg_file) + "_translated");
    convertDawgToWordList(unicharset_file, dawg_file, tmp_wordlist.path.string().c_str());

    // Load both word lists to tesseract::trie

    tesseract::CheckSharedLibraryVersion();
    tesseract::Classify *classify = new tesseract::Classify();
    tprintf("Loading unicharset from '%s'\n", unicharset_file);

    if (!classify->getDict().getUnicharset().load_from_file(unicharset_file)) {
        tprintf("Failed to load unicharset from '%s'\n", unicharset_file);
        delete classify;
        return;
    }

    // First add original word list.
    const UNICHARSET &unicharset = classify->getDict().getUnicharset();
    tesseract::Trie::RTLReversePolicy reverse_policy =
                tesseract::Trie::RRP_DO_NO_REVERSE;

    tesseract::Trie trie(
            // the first 3 arguments are not used in this case
            tesseract::DAWG_TYPE_WORD, "", SYSTEM_DAWG_PERM,
            unicharset.size(), classify->getDict().dawg_debug_level);

    tprintf("Reading word list from '%s'\n", wordlist_file);
    if (!trie.read_and_add_word_list(wordlist_file, unicharset,
                                     reverse_policy)) {
        tprintf("Failed to add word list from '%s'\n", wordlist_file);
        delete classify;
        return;
    }

    // Next add modified word list.
    tprintf("Reading word list from '%s'\n", tmp_wordlist.path.string().c_str());
    if (!trie.read_and_add_word_list(tmp_wordlist.path.string().c_str(), unicharset,
                                         reverse_policy)) {
        tprintf("Failed to add word list from '%s'\n", tmp_wordlist.path.string().c_str());
        delete classify;
        return;
    }


    tprintf("Reducing Trie to SquishedDawg\n");
    tesseract::SquishedDawg *dawg = trie.trie_to_dawg();
    if (dawg != nullptr && dawg->NumEdges() > 0) {
        tprintf("Writing squished DAWG to '%s'\n", dawg_file);
        dawg->write_squished_dawg(dawg_file);
    } else {
        tprintf("DAWG is empty, skip producing the output file\n");
    }

    delete dawg;
    delete classify;

}


/**
 * Converts a text formatted word list file into a given DAWG file. Replaces original DAWG if it exists.
 * Note: The correct unicharset file must be provided or results will be gibberish.
 *
 * @param unicharset_file - Unicharset file contained within target traineddata model file.
 * @param wordlist_file - Path to given word list text file.
 * @param dawg_file  - Path of DAWG output file.
 */
void MPF_Model_Updater::copyWordListOverDawg(const char *unicharset_file,
                                             const char *wordlist_file,
                                             const char *dawg_file) {

    boost::filesystem::remove_all(dawg_file);
    convertWordListToDawg(unicharset_file, wordlist_file, dawg_file);
}

/**
 * Helper function checks for *.traineddata files within target directory.
 *
 * @param model_dir - Path to model files.
 * @param model_set - Add model filename to given set of models.
 * @param is_empty  - Reports if directory is empty or not.
 */
void MPF_Model_Updater::checkModels(const char *model_dir,
                                    std::set<std::string> &model_set,
                                    bool &is_empty) {
    boost::filesystem::directory_iterator iter(model_dir);
    boost::filesystem::directory_iterator empty;

    if (iter == empty) {
        is_empty = true;
        return;
    }

    is_empty = false;

    while(iter != empty){
        if (boost::filesystem::is_directory(iter->path())) {
            iter++;
            continue;
        }
        std::string lang = iter->path().stem().string();

        model_set.insert(lang);
        iter++;
    }
}

/**
 * Inspect user provided dict directory to see if any files are present.
 * @param dict_dir - Directory of model files to add to existing models.
 * @param original_models - Original models in target directory.
 * @param updated_models - Pre-existing models in output directory.
 * @param lang_dict_map - Stores new files and target models to be updated.
 * @return - true if files are available for updated models.
 */
bool MPF_Model_Updater::checkDictDir(const char *dict_dir,
                                     std::set<std::string> &original_models,
                                     std::set<std::string> &updated_models,
                                     std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                             std::vector<std::string>>> &lang_dict_map) {

    boost::filesystem::directory_iterator dict_dir_iter(dict_dir);
    boost::filesystem::directory_iterator empty;

    if (dict_dir_iter == empty) {
        printf("Warning returning early due to empty models/dict directory.\n");
        return false;
        //return updated_models;
    }

    while(dict_dir_iter != empty){
        if (boost::filesystem::is_directory(dict_dir_iter->path())) {
            dict_dir_iter++;
            continue;
        }

        std::string lang;
        if (boost::algorithm::contains(dict_dir_iter->path().string(), ".txt")) {
            lang = dict_dir_iter->path().stem().stem().string();
        } else {
            lang = dict_dir_iter->path().stem().string();
        }
        // If language model does not exist or has already been updated, skip associated language files.
        if (original_models.find(lang) == original_models.end() ||
            updated_models.find(lang) != updated_models.end()) {
            printf("Skipping lang: %s\n", lang.c_str());
            dict_dir_iter++;
            continue;
        }

        // Add new language here, skip languages that are missing from original models or present in updated models.
        if (lang_dict_map.find(lang) == lang_dict_map.end()){
            printf("Adding lang: %s\n", lang.c_str());
            std::pair<std::string, std::pair<std::vector<std::string>, std::vector<std::string>>> lang_files(
                    lang,
                    std::pair<std::vector<std::string>, std::vector<std::string>>());
            lang_dict_map.insert(lang_files);
        }

        // Add file, either to the DAWG list or the non-DAWG list.
        if (dict_dir_iter->path().has_extension() &&
            (boost::algorithm::contains(dict_dir_iter->path().extension().string(), "dawg") ||
             boost::algorithm::contains(dict_dir_iter->path().extension().string(), "txt"))) {

            printf("Adding model dictionary file (text, DAWG): %s\n",  dict_dir_iter->path().string().c_str());
            lang_dict_map.at(lang).second.push_back(dict_dir_iter->path().filename().string());
        } else {
            printf("Adding Non-DAWG model file: %s\n",  dict_dir_iter->path().string().c_str());
            lang_dict_map.at(lang).first.push_back(dict_dir_iter->path().filename().string());
        }
        dict_dir_iter++;
    }
    return true;
}


/**
 * Updates all models in target directory with respective model files.
 *
 * @param model_dir - Directory of models to be updated.
 * @param dict_dir - Directory of model files to add to existing models.
 * @param updated_model_dir - Output directory for updated models.
 * @param force_update - If true (default) overwrite and update all models.
 *                       If false, skips updating models already present in updated_model_dir.
 *
 * @return - Vector of updated models.
 */
std::set<std::string> MPF_Model_Updater::updateLanguageFiles(const char *model_dir,
                                                             const char *dict_dir,
                                                             const char *updated_model_dir,
                                                             bool force_update,
                                                             bool replace_dawgs) {

    // First isolate every language present in dict_dir. Bundle up all associated files and DAWGs with each model.
    std::unordered_map<std::string, std::pair<std::vector<std::string>, std::vector<std::string>>> lang_dict_map;
    std::set<std::string> original_models, updated_models;
    bool is_empty = false;

    // Load existing language models.
    // If updated_models_dir does not exist, create it.
    if (!boost::filesystem::is_directory(updated_model_dir)) {
        boost::filesystem::remove_all(updated_model_dir);
        boost::filesystem::create_directory(updated_model_dir);
    } else if (!force_update) {
        // If updates are forced, then ignore existing updated models, and overwrite them later.
        checkModels(updated_model_dir, updated_models, is_empty);
    }
    checkModels(model_dir, original_models, is_empty);

    // Inspect user provided dict directory to see if any files are present.
    // Loaded files into lang_dict_map.
    if (is_empty || !checkDictDir(dict_dir, original_models, updated_models, lang_dict_map)) {
        return updated_models;
    }

    // Then create a temporary directory to store all model files.
    TempRandomDirectory temp_dir(updated_model_dir);

    // Also create a temporary directory to store intermediate DAWG files.
    boost::filesystem::path tmp_dawg_dir_path = boost::filesystem::path(updated_model_dir);
    TempRandomDirectory tmp_dawg_dir(tmp_dawg_dir_path.string(), "/tmp-dawg-dir-%");

    for (auto &it: lang_dict_map) {
        printf("\nProcessing %s\n", it.first.c_str());
        std::vector<std::string> updated_files;
        std::vector<char*> updated_files_c_str;

        // Unpack each model file into temp directory.
        boost::filesystem::path path_to_model = boost::filesystem::path(model_dir) / (it.first + ".traineddata");
        boost::filesystem::path output_prefix = temp_dir.path / it.first;

        MPF_Model_Updater::extractLangModel(path_to_model.string(), output_prefix.string());

        boost::filesystem::path path_tmp_model = output_prefix.replace_extension(".traineddata");
        boost::filesystem::path path_out_model = boost::filesystem::path(updated_model_dir) /
                                                 (it.first + ".traineddata");

        // Copy model into temp directory as well.
        boost::filesystem::copy_file(path_to_model,
                                     path_tmp_model,
                                     boost::filesystem::copy_option::overwrite_if_exists);

        // First copy over each non-DAWG file.
        for (std::string &file: it.second.first) {
            boost::filesystem::path src = boost::filesystem::path(dict_dir) / file;
            boost::filesystem::path dst = temp_dir.path / file;
            boost::filesystem::remove_all(dst);
            boost::filesystem::copy_file(src, dst);

            updated_files.push_back(dst.string());
            updated_files_c_str.push_back(&updated_files.back()[0]);
        }

        // Then replace each DAWG file in tmp directory with its counterpart.
        // Check based whether LSTM version should be used.
        for (std::string &file: it.second.second) {
            bool text_format = false;

            std::string target_file = file;

            if (boost::filesystem::extension(file) == ".txt") {
                text_format = true;
                target_file = boost::filesystem::path(file).replace_extension("").string();
                printf("Updating DAWG file %s with text-based word list %s\n", target_file.c_str(), file.c_str());
            } else {
                printf("Combining model DAWG with given DAWG: %s\n", file.c_str());
            }


            // Get proper unicharset file version (legacy, LSTM).
            std::string unicharset_file = ".unicharset";
            if (boost::algorithm::contains(file,"lstm")) {
                unicharset_file = ".lstm-unicharset";
            }
            boost::filesystem::path unichar_path = temp_dir.path / (it.first + unicharset_file);


            // Convert to text if word list is in DAWG format.
            boost::filesystem::path src_wordlist = boost::filesystem::path(dict_dir) / file;
            if (!text_format){

                boost::filesystem::path translated_wordlist = tmp_dawg_dir.path / (file + ".txt");

                convertDawgToWordList(unichar_path.string().c_str(),
                                      src_wordlist.string().c_str(),
                                      translated_wordlist.string().c_str());
                src_wordlist = translated_wordlist;
            }

            boost::filesystem::path dst_dawg = temp_dir.path / target_file;
            updated_files.push_back(dst_dawg.string());
            updated_files_c_str.push_back(&updated_files.back()[0]);


            
            if (!replace_dawgs) {
                addWordListToDawg(unichar_path.string().c_str(),
                                  src_wordlist.string().c_str(),
                                  dst_dawg.string().c_str());
            } else {
                copyWordListOverDawg(unichar_path.string().c_str(),
                                     src_wordlist.string().c_str(),
                                     dst_dawg.string().c_str());
            }

        }

        // Finally update model in tmp directory with all specified files, then copy over to updated directory.
        MPF_Model_Updater::updateLanguageModel(path_tmp_model.string().c_str(),
                                               updated_files_c_str.data(),
                                               updated_files_c_str.size());
        boost::filesystem::remove(path_out_model);
        boost::filesystem::copy_file(path_tmp_model, path_out_model);
    }
    return updated_models;
}

int main(int argc, char **argv) {
    tesseract::CheckSharedLibraryVersion();

    int i;
    tesseract::TessdataManager tm;
    if (argc > 1 && (!strcmp(argv[1], "-v") || !strcmp(argv[1], "--version"))) {
        printf("%s\n", tesseract::TessBaseAPI::Version());
        return EXIT_SUCCESS;
    } else if (argc == 2) {
        printf("Combining tessdata files\n");
        STRING lang = argv[1];
        char* last = &argv[1][strlen(argv[1])-1];
        if (*last != '.')
            lang += '.';
        STRING output_file = lang;
        output_file += kTrainedDataSuffix;
        if (!tm.CombineDataFiles(lang.string(), output_file.string())) {
            printf("Error combining tessdata files into %s\n",
                   output_file.string());
        } else {
            printf("Output %s created successfully.\n", output_file.string());
        }
    } else if (argc >= 4 && (strcmp(argv[1], "-dw") == 0)) {
        convertDawgToWordList(argv[2], argv[3], argv[4]);
    } else if (argc >= 4 && (strcmp(argv[1], "-wd") == 0)) {
        convertWordListToDawg(argv[2], argv[3], argv[4]);
    } else if (argc >= 4 && (strcmp(argv[1], "-e") == 0 )) {
        extractLangModel(std::string(argv[2]), std::string(argv[3]));
    } else if (argc >= 4 && (strcmp(argv[1], "-u") == 0 )) {
        updateLanguageFiles(argv[2], argv[3], argv[4]);
    } else if (argc >= 4 && (strcmp(argv[1], "-ur") == 0 )) {
        updateLanguageFiles(argv[2], argv[3], argv[4], true, true);
    } else if (argc >= 4 && (strcmp(argv[1], "-c") == 0 )) {
        combineWordLists(argv[2], argv[3], argv[4]);
    } else if (argc >= 4 && strcmp(argv[1], "-o") == 0) {
        updateLanguageModel(argv[2], argv+3, argc-3);
    } else {

        printf("\nUsage for updating all models in target directory:\n"
               "    %s -u <ORIGINAL_MODELS_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>\n\n",  argv[0]);

        printf("Usage for updating all models in target directory via replacement:\n"
               "    %s -ur <ORIGINAL_MODELS_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>\n\n",  argv[0]);

        printf("Usage for combining tessdata components into a single model:\n"
               "    %s <INPUT_LANGUAGE_PATH_PREFIX>\n\n",  argv[0]);

        printf("Usage for extracting all tessdata components:\n"
               "    %s -e <TRAINEDDATA_FILE> <OUTPUT_LANGUAGE_PATH_PREFIX>\n\n",  argv[0]);

        printf("Usage for overwriting tessdata components:\n"
               "    %s -o <TRAINEDDATA_FILE> [INPUT_COMPONENT_FILE...]\n\n",  argv[0]);

        printf("Usage for converting DAWG model files to word list text files:\n"
               "    %s -dw <TRAINEDDATA_UNICHARSET_FILE> <TRAINEDDATA_DAWG_FILE> <OUTPUT_WORD_LIST_FILE>\n\n",  argv[0]);

        printf("Usage for converting word list text files back to DAWG files:\n"
               "    %s -wd <TRAINEDDATA_UNICHARSET_FILE> <WORD_LIST_FILE> <OUTPUT_DAWG_FILE>\n\n",  argv[0]);

        printf("Usage for combining two text-formatted word lists together:\n"
               "    %s -c <WORD_LIST_FILE_1> <WORD_LIST_FILE_2> <OUTPUT_WORD_LIST_FILE>\n\n",  argv[0]);

        printf("Please refer to TesseractOCRTextDetection DICTIONARIES.md for more examples and details.\n\n");
        return 1;
    }
    tm.Directory();
    return EXIT_SUCCESS;
}
