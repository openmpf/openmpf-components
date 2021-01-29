# Inspecting and Updating Tessdata Models With Custom Word Dictionaries

You may inspect and update your existing Tesseract `*.traineddata` models with
the `tessdata_model_updater` app. This app is installed onto the Tesseract
Docker component image.

If you already have a Docker container running, you can use the app as follows:

`docker exec <MPF_TESSERACT_CONTAINER> /opt/mpf/tessdata_model_updater <MODEL_UPDATER_COMMAND>`

Alternatively, you can run a container to execute a single command:

`docker run --rm [-v LOCAL_PATH:CONTAINER_PATH] --entrypoint /opt/mpf/tessdata_model_updater <MPF_TESSERACT_IMAGE> <MODEL_UPDATER_COMMAND>`

For example:

`docker run --rm -v "$(pwd)":/work --entrypoint /opt/mpf/tessdata_model_updater openmpf_tesseract_ocr_text_detection:latest -c /work/eng.word-dawg.txt /work/updates.txt /work/combined.txt`

Below is a summary of the available model updater app commands. You may want to
skip ahead to the [Automated Walkthrough](#automated-walkthrough) section for an
example of a typical use case.


## Extract and Inspect Files

### Extract `*.traineddata` Model

Extract a given model's component files (dictionaries, unicharset, etc.) into
target output path:

`./tessdata_model_updater -e <TRAINEDDATA_FILE> <OUTPUT_LANGUAGE_PATH_PREFIX>`

The `OUTPUT_LANGUAGE_PATH_PREFIX` is applied to all extracted files.

For example, to extract out the `eng.*-dawg` files from the `eng.traineddata`
model (as well as all of the other component files) into another directory:

`./tessdata_model_updater -e eng.traineddata extracted_model_dir/eng`

This command will fail if `extracted_model_dir` does not exist. You may need to
create it first.

### Convert `*-dawg` File to Word List

Convert an extracted model `*-dawg` dictionary file to a text-formatted
word list:

`./tessdata_model_updater -dw <TRAINEDDATA_UNICHARSET_FILE> <TRAINEDDATA_DAWG_FILE> <OUTPUT_WORD_LIST_FILE>`

Example:

`./tessdata_model_updater -dw eng.unicharset eng.word-dawg eng.word-dawg.txt`

Note that the `eng.unicharset` file was previously extracted from the existing
`eng.traineddata` model using the `-e` command, as described
[here](#extract-traineddata-model).

Some language models contain legacy and LSTM `unicharset` files (ex.
`eng.unicharset`, `eng.lstm-unicharset`).

Please ensure that you use the correct `unicharset` file.
- Example: `eng.lstm-unicharset` pairs with `eng.lstm-*-dawg` files.
- Example: `eng.unicharset` pairs with `eng.*-dawg` files.

## Create and Update Files

### Combine Word Lists

Combine two text formatted word lists together:

`./tessdata_model_updater -c <WORD_LIST_FILE_1> <WORD_LIST_FILE_2> <OUTPUT_WORD_LIST_FILE>`

Each word list file provided should contain a newline-separated list of words.

Example:

`./tessdata_model_updater -c eng.word-dawg.txt eng.updated.word-dawg.txt eng.combined.word-dawg.txt`

### Convert Word List into `*-dawg` File

Convert a given word list into a model DAWG format:

`./tessdata_model_updater -wd <TRAINEDDATA_UNICHARSET_FILE> <WORD_LIST_FILE> <OUTPUT_DAWG_FILE>`

Example:

`./tessdata_model_updater -wd eng.unicharset eng.word-dawg.txt eng.word-dawg`

Note that the `eng.unicharset` file was previously extracted from the existing
`eng.traineddata` model using the `-e` command, as described
[here](#extract-traineddata-model).

Some language models contain legacy and LSTM `unicharset` files (ex.
`eng.unicharset`, `eng.lstm-unicharset`).

Please ensure that you use the correct `unicharset` file.
- Example: `eng.lstm-unicharset` pairs with `eng.lstm-*-dawg` files.
- Example: `eng.unicharset` pairs with `eng.*-dawg` files.

In addition, you must provide the word list file as a text-formatted list of
words separated by newlines.

NOTE: Please be aware that a small fraction (~1%) of words may be filtered out
during the initial word list to DAWG conversion step. Subsequent conversation
between DAWG and text formats will retain all leftover words.

### Overwrite `*.traineddata` Model with New Files

Overwrite a Tesseract model with new word dictionaries or other model
components:

`./tessdata_model_updater -o <TRAINEDDATA_FILE> [INPUT_COMPONENT_FILE...]`

You can replace any components in a model (ex. `eng.word-dawg`,
`eng.unicharset`) by listing files that share the same extension as the
components they should replace.

Example:

`./tessdata_model_updater -o eng.traineddata eng.new.unicharset eng.new.word-dawg`

This will replace the existing `eng.unicharset` and `eng.word-dawg` file in the
`eng.traineddata` model.

If more than one file is specified with the same extension (ex. two
`*.word-dawg` files), then only the last one provided is used.

### Create `*.traineddata` Model from Parts

Combine extracted model components back into a single `*.traineddata` model:

`./tessdata_model_updater <INPUT_LANGUAGE_PATH_PREFIX>`

For example, to generate a `eng.traineddata` model from `eng.*` files inside of
directory called `tessdata_extracted_files`:

`./tessdata_model_updater tessdata_extracted_files/eng`

### Update Multiple Models with New Files

Automatically update all models in target directory with a given set of new
model component files:

`./tessdata_model_updater -u <ORIGINAL_MODELS_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>`

This command will fail if `OUTPUT_MODELS_DIR` does not exist. You may need to
create it first.

This command automates many of the previous commands to update all model files
in a given target directory. **Specifically, DAWG components will be updated by
adding words to their existing word list. Non-DAWG components will simply be
replaced with the newer version.**

Each updated file in `UPDATED_COMPONENT_FILES_DIR` must have the exact same name
as the target component inside of the `*.traineddata` model. The only exception
is when updating word dictionaries two formatting options are allowed:

- You may provide a `<LANGUAGE>.*-dawg` file in the Tesseract DAWG format.
(Use the `-dw` conversion command on existing text-based word lists to generate
  these files, as explained [here](#convert-word-list-into--dawg-file).)
    - Example: Providing `fra.word-dawg` in `UPDATED_COMPONENT_FILES_DIR` will
    update the `fra.word-dawg` model component.

- You may provide a `<LANGUAGE>.*-dawg.txt` as a text-formatted
newline-separated list of words. The word list must match the name of the
target `<LANGUAGE>.*-dawg` component with an extra `.txt` extension at the end.
    - Example: Providing `eng.word-dawg.txt` in `UPDATED_COMPONENT_FILES_DIR`
    will update the `eng.word-dawg` model component.

Refer to the [Automated Walkthrough](#automated-walkthrough) section for an
example of a typical use case.

### Update Multiple Models with New Files via Replacement

The `-ur` command is similar to the above `-u` command, but instead new DAWG
components will be generated rather than adding to the word list of the existing
ones:

`./tessdata_model_updater -ur <ORIGINAL_MODELS_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>`


# Walkthroughs for Updating Existing Model Dictionaries

The following walkthroughs provide steps for updating the `eng.traineddata` file
with new dictionary words. They both accomplish the same goal.

The manual walkthrough requires running multiple commands while the automated
walkthrough requires running a single command. You may wish to follow the manual
walkthough to familiarize yourself with the process, or debug issues.

## Manual Walkthrough

Given:
- `eng.traineddata`: Target model
- `new-english-wordlist.txt`: Target words to add to target model
- `<TEMP_DIR>`: Temporary directory for storing intermediate model files

Run the following command to extract out word list `*-dawg` files and `unicharset`
files:

`./tessdata_model_updater -e eng.traineddata <TEMP_DIR>/eng`

Run the following command to convert the English word dictionary to text format:

`/tessdata_model_updater -dw <TEMP_DIR>/eng.unicharset <TEMP_DIR>/eng.word-dawg <TEMP_DIR>/eng.word-dawg.txt`

Run the following command to update the English word dictionary with new words:

`./tessdata_model_updater -c new-english-wordlist.txt <TEMP_DIR>/eng.word-dawg.txt <TEMP_DIR>/eng.word-dawg-updated.txt`

Run the following command to convert the updated English word dictionary back
into the `.word-dawg` file:

`./tessdata_model_updater -wd new-english-wordlist.txt <TEMP_DIR>/eng.word-dawg-updated.txt <TEMP_DIR>/eng.word-dawg`

Run the following command to overwrite the model's existing word dictionary with
the updated dictionary:

`/tessdata_model_updater -o eng.traineddata <TEMP_DIR>/eng.word-dawg`

## Automated Walkthrough

If you have multiple language models and word dictionaries, you can also use the
automatic model update commands (`-u` or `-ur`) as follows.

Given:
- `<ORIGINAL_TESSDATA_MODELS_DIR>`: Contains the original `*.traineddata` models
- `<UPDATED_COMPONENT_FILES_DIR>`: Contains sets of `<LANGUAGE>.*-dawg` and
  other model files to add to existing models
- `<OUTPUT_MODELS_DIR>`: Will contain the updated `*.traineddata` output models.
  You can set this equal to `ORIGINAL_TESSDATA_MODELS_DIR` to update all models directly.

Place files in `UPDATED_COMPONENT_FILES_DIR` that will be used to update the
existing `*.traineddata` files in `ORIGINAL_TESSDATA_MODELS_DIR`. You don't need
to include all of the files, only those that are different (updated) from the
originals.

Each updated file in `UPDATED_COMPONENT_FILES_DIR` must have the exact same name
as the target component inside of the `*.traineddata` model. The only exception
is when updating word dictionaries two formatting options are allowed:

- You may provide a `<LANGUAGE>.*-dawg` file in the Tesseract DAWG format.
(Use the `-dw` conversion command on existing text-based word lists to generate
  these files, as explained [here](#convert-word-list-into--dawg-file).)
    - Example: Providing `fra.word-dawg` in `UPDATED_COMPONENT_FILES_DIR` will
    update the `fra.word-dawg` model component.

- You may provide a `<LANGUAGE>.*-dawg.txt` as a text-formatted
newline-separated list of words. The word list must match the name of the
target `<LANGUAGE>.*-dawg` component with an extra `.txt` extension at the end.
    - Example: Providing `eng.word-dawg.txt` in `UPDATED_COMPONENT_FILES_DIR`
    will update the `eng.word-dawg` model component.

Run the following command to add all `<LANGUAGE>.*-dawg` files to their
respective language models:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODELS_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>`

Run the following command to add all `script/<LANGUAGE>.*-dawg` files to their
respective script models:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODELS_SCRIPT_DIR> <UPDATED_COMPONENT_FILES_DIR> <OUPUT_MODELS_SCRIPT_DIR>`

This is typically equivalent to:

`tessdata_model_updater -u <ORIGINAL_TESSDATA_MODELS_DIR>/script <UPDATED_COMPONENT_FILES_DIR> <OUTPUT_MODELS_DIR>/script`

(If you wish to replace existing word dictionaries with a completely new set of
words, run the above commands with `-ur` instead of `-u`.)

Please note that any updated `unicharset` files provided in
`UPDATED_COMPONENT_FILES_DIR` will be used during the model dictionary updates.

For example, if both `eng.word-dawg.txt` and `eng.unicharset` are in
`UPDATED_COMPONENT_FILES_DIR` in the above example commands:
- Then `eng.unicharset` will first replace the default `unicharset` file for
  `eng.traineddata`.
- Afterwards `eng.word-dawg.txt` will be added to the `eng.traineddata` word
  dictionary using the updated unicharaset file as reference during the word list
  to DAWG conversion step.

NOTE: Please be aware that the word list to DAWG conversion step may skip/filter
out a small subset of select words when generating the initial `*-dawg` files.
Subsequent conversion between DAWG and text formats will keep remaining words.
