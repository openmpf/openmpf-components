# Inspecting and Updating Tessdata Models With Custom Word Dictionaries

Users may inspect and update their existing Tesseract `*.traineddata` models with the `tessdata_model_updater` app.

This app is installed onto the Tesseract Docker component image and can be run as follows:

`docker exec -it <MPF_TESSERACT_CONTAINER> /opt/mpf/tessdata_model_updater <MODEL_UPDATER_COMMAND>`

Below is a summary of the available model updater commands.

## Extract and Inspect Files

### Extract `*.traineddata` Model

Extract a given model's component files (dictionaries, unicharset, etc.) into target output path:

`./tessdata_model_updater -e traineddata_file output_path_prefix`

The `output_path_prefix` is applied across all extracted files.

For example, to extract out the `eng.*-dawg` files from `eng.traineddata` model into another directory:

`./tessdata_model_updater -e eng.traineddata extracted_model_dir/eng`

### Convert `*-dwag` File to Word List

Convert extracted model `*-dawg` dictionary files to text formatted wordlist:

`./tessdata_model_updater -dw traineddata_unicharset_file traineddata_dawg_file output_text_file`

Note that the `unicharset` file was previously extracted from an existing `trainedata` model using the `-e` command,
as described in the above section.

Some language models contain legacy and LSTM `unicharset` files (ex. `eng.unicharset`, `eng.lstm-unicharset`).

Please ensure that each `*-dawg` file is paired with the correct `unicharset` file.
- Example: `eng.lstm-unicharset` pairs with `eng.lstm-*-dawg` files.
- Example: `eng.unicharset` pairs with `eng.*-dawg` files.

## Create and Update Files

### Combine Word Lists

Combine two text formatted wordlists together:

`./tessdata_model_updater -c wordlist_file_1 wordlist_file_2 output_word_list`

Each file provided should contain a newline-separated list of words.

### Convert Word List into `*-dwag` File

Convert a given wordlist into a model DAWG format:

`./tessdata_model_updater -wd traineddata_unicharset_file wordlist_text_file output_traineddata_dawg_file`

Note that the `unicharset` file was previously extracted from an existing `trainedata` model using the `-e` command,
as described in the above section.

Some language models contain legacy and LSTM `unicharset` files (ex. `eng.unicharset`, `eng.lstm-unicharset`).

Please ensure that each `*-dawg` file is paired with the correct `unicharset` file.
- Example: `eng.lstm-unicharset` pairs with `eng.lstm-*-dawg` files.
- Example: `eng.unicharset` pairs with `eng.*-dawg` files.

In addition, users must provide the wordlist file as a text-formatted list of words separated by newlines.

NOTE: Please be aware that a small fraction (~1%) of words may be filtered out during the initial wordlist
to DAWG conversion step. Subsequent conversation between DAWG and text formats will retain all leftover words.

### Overwrite `*.traineddata` Model with New Files

* Overwrite a Tesseract model with new word dictionaries or other model components:

`./tessdata_model_updater -o traineddata_file [input_component_file...]`

Users can effectively replace any components in a model (ex. `eng.word-dawg`, `eng.unicharset`) by adding
replacement files named after the target component extension.

Example:

`./tessdata_model_updater -o eng.traineddata eng.unicharset eng.word-dawg`

This will replace the existing `eng.unicharset` and `eng.word-dawg` file in the `eng.traineddata` model.

If more than one file is specified with the same extension (ex. two `*.word-dawgs`), then only the last one
provided is used.

### Create `*.traineddata` Model from Parts

Combine extracted model components back into a single `*.traineddata` model:

`./tessdata_model_updater language_data_path_prefix`

For example, to generate a `eng.traineddata` model from `eng.*` files inside of directory called `tessdata_extracted_files`:

`./tessdata_model_updater tessdata_extracted_files/eng`

### Update All Models with New Files

Automatically update all models in target directory with a given set of new model component files:

`./tessdata_model_updater -u original_models_dir updated_model_files_dir output_updated_models_dir`

This command is effectively automates many of the previous commands to update all model files in a given target
directory.

When updating DAWG dictionaries, the user provided files will be appended to the existing `.*-dawg` wordlists.

For ease of use, users may provide dictionary updates in either `.*-dawg` or `.txt` formats.

The given wordlists must match the target model component names and file extensions, with text files having an additional `.txt` extension.

Example: Providing `eng.word-dawg.txt` in `updated_model_files_dir` will update the `eng.word-dawg` model component.

Example: Providing `fra.word-dawg` in `updated_model_files_dir` will update the `fra.word-dawg` model component.

Non-DAWG files will be simply replaced with the newer version.

To replace all model components (including DAWG files), then run the following command instead:

`./tessdata_model_updater -u original_models_dir updated_model_files_dir output_updated_models_dir`

Please see the [Automated Walkthrough](#automated-walkthrough) section for more details.

# Walkthroughs for Updating Existing Model Dictionaries

The following walkthroughs provide steps for updating the `eng.traineddata` file with new dictionary words.
They both accomplish the same goal.

The manual walkthrough requires running multiple commands while the automated walkthrough requires running a single
command. You may with to follow the manual walkthough to familiarize yourself with the process, or debug issues.

## Manual Walkthrough

Given:
- `eng.traineddata`: Target model
- `new-english-wordlist.txt`: Target words to add to target model
- `<TEMP_DIR>`: Temporary directory for storing intermediate model files

Run the following command to extract out wordlists `dawgs` and `unicharset` files:

`./tessdata_model_updater -e eng.traineddata <TEMP_DIR>/eng`

Run the following command to convert the English word dictionary to text format:

`/tessdata_model_updater -dw <TEMP_DIR>/eng.unicharset <TEMP_DIR>/eng.word-dawg <TEMP_DIR>/eng.word-dawg.txt`

Run the following command to update the English word dictionary with new words:

`./tessdata_model_updater -c new-english-wordlist.txt <TEMP_DIR>/eng.word-dawg.txt <TEMP_DIR>/eng.word-dawg-updated.txt`

Run the following command to convert the updated English word dictionary back into the `.word-dawg` file:

`./tessdata_model_updater -wd new-english-wordlist.txt <TEMP_DIR>/eng.word-dawg-updated.txt <TEMP_DIR>/eng.word-dawg`

Run the following command to overwrite the component's existing word dictionary with the updated dictionary:

`/tessdata_model_updater -o eng.traineddata <TEMP_DIR>/eng.word-dawg`

## Automated Walkthrough

If users have multiple language models and word dictionaries, they can also use the automatic model update
commands (`-u` or `-ur`) as follows.

Given:
- `<ORIGINAL_TESSDATA_MODEL_DIR>`: Contains the original `*.traineddata` models
- `<UPDATED_DICT_FILES_DIR>`: Contains sets of `<LANGUAGE>.*-dawg` and other model files to add to existing models
- `<UPDATED_MODEL_DIR>`: Will contain the updated `*.traineddata` output models.
    You can set this equal to `ORIGINAL_TESSDATA_MODEL_DIR` to update all models directly.

Place files in `UPDATED_DICT_FILES_DIR` that will be used to update the existing `*.traineddata` files in `ORIGINAL_TESSDATA_MODEL_DIR`. You don't need to include all of the files, only those that are different (updated) from the originals.

Each updated file in `UPDATED_DICT_FILES_DIR` must have the exact same name as the target component inside of the `*.traineddata` model. The only exception is when updating word dictionaries two formatting options are allowed:
- You may provide a `<LANGUAGE>.*-dawg` file in the Tesseract DAWG format. (Use the `-dw` conversion command on existing text-based wordlists to generate these files.)
- You may provide a `<LANGUAGE>.*-dawg.txt` as a text-formatted newline-separated list of words. The wordlist must match the name of the target `<LANGUAGE>.*-dawg` component with an extra `.txt` extension at the end.

Run the following command to add all `<LANGUAGE>.*-dawg` files to their respective languages:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODEL_DIR> <UPDATED_DICT_FILES_DIR> <UPDATED_MODEL_DIR>`

Run the following command to add all `script/<LANGUAGE>.*-dawg` files to their respective scripts:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODEL_SCRIPT_DIR> <UPDATED_DICT_FILES_DIR> <UPDATED_MODEL_SCRIPT_DIR>`

This is typically equivalent to:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODEL_DIR>/script <UPDATED_DICT_FILES_DIR> <UPDATED_MODEL_DIR>/script`

If users wish to replace existing word dictionaries with a completely new set of words, run the following commands instead:

- `tessdata_model_updater -ur <ORIGINAL_TESSDATA_MODEL_DIR> <UPDATED_DICT_FILES_DIR> <UPDATED_MODEL_DIR>`

- `tessdata_model_updater -ur <ORIGINAL_TESSDATA_MODEL_SCRIPT_DIR> <UPDATED_DICT_FILES_DIR> <UPDATED_MODEL_SCRIPT_DIR>`

Please note that any updated `unicharset` files provided in `UPDATED_DICT_FILES_DIR` will be used during the model dictionary updates.

For example, if both `eng.word-dawg.txt` and `eng.unicharset` are in `UPDATED_DICT_FILES_DIR` in the above example commands:
- Then `eng.unicharset` will first replace the default `unicharset` file for `eng.traineddata`.
- Afterwards `eng.word-dawg.txt` will be added to the `eng.traineddata` word dictionary using the updated unicharaset file as reference during the wordlist to DAWG conversion step.

Also note that the wordlist to DAWG conversion step may skip/filter out a small subset of select words
when generating the initial `*-dawg` files. Subsequent conversion between DAWG and text formats will keep remaining words.
