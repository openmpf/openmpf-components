# Overview

This repository contains source code and model data for the OpenMPF Tesseract
OCR text detection component.

The component extracts text found in an image, reported as a single track detection.
PDF documents can also be processed with one track detection per page. The first page
corresponds to the detection property PAGE_NUM = 1. For debugging purposes, images converted
from documents are stored in a temporary job directory under
`plugin/TesseractOCR/tmp-[job-id]-[random tag]`. This directory is removed when the job completes successfully.

Please refer to https://imagemagick.org/script/formats.php for support of other document file formats.

Users may set the language of each track using the TESSERACT_LANGUAGE parameter
as well as adjust image preprocessing settings for text extraction.

For processing English text, users can enable filters (THRS_FILTER, HIST_FILTER)
to eliminate gibberish detections from a given scene. All text extracted from
an image can also be tagged using regex and keyword tags in a given json file.
For keyword tagging, users can provide either words or phrases
(ex. "bank-tag: [money, bank of america, etc.]"). Phrases must contain
words separated by white-space. For more complex pattern matching, use regex tags
instead. Both forms of tagging are case-insensitive.


By default the json tagging file is located in the config folder as text-tags.json,
however users can provide an alternate full path to a tagging file of their choice.
English and foreign text tags following UTF-8 encoding are supported.


Furthermore, language models supported by Tesseract are stored by default in
`$MPF_HOME/plugins/TesseractOCRTextDetection/tessdata/tessdata` directory and script models
are stored in `tessdata/script`.
For development and testing purposes outside of the workflow manager, the tessdata directory is located in
`TesseractOCRTextDetection/plugin-files/tessdata`. Users can set a new tessdata directory by
modifying the TESSDATA_DIR job property. Additional tessdata models can then be added to the specified
tessdata folder to expand supported languages and scripts.

Each language module follows ISO 639-2 designations, with character variations
of a language (ex. uzb_cyrl) also supported. Users must enter the same designation
to enable the corresponding language detection (ex. TESSERACT_LANGUAGE = "deu"
for German text). Users will be warned when a given language is not supported when the
corresponding language module cannot be located in this directory.

Each script module is contained within the `tessdata/script` directory with the
first letter of its type capitalized (ex. `tessdata/script/Latin.traineddata`). Users will need to
specify the `script/` path followed by the full name of the script being processed.
(ex. TESSERACT_LANGUAGE='script/Latin' will enable Latin script text extraction).

Users can set ENABLE_OSD to true to enable automatic orientation and script detection in
place of a prespecified language. Setting ENABLE_OSD_TRACK_REPORT to true will report script and
orientation results as an additional separate track. Based on MIN_ORIENTATION_CONFIDENCE and MIN_SCRIPT_CONFIDENCE
thresholds the component will then use the detected script and text orientation (0, 90, 180, and 270 degree rotations)
instead of default language and orientation (no rotation). By default only one detected script will be reported and
used, however users can set MAX_SCRIPTS to 2 or greater to predict and process multiple scripts. Users must
also set the MIN_SECONDARY_SCRIPT_THRESHOLD value for multi-script detection to allow for score comparisons between
the primary script and secondary script predictions.

There are two options to run multiple prespecified languages/scripts. Users can separate each
specified language and script using the '+' delimiter to run multiple models
together in one track and ',' to run them as separate tracks. Delimiters
can also be combined for separate multilingual tracks. Lastly, users can set MAX_TEXT_TRACKS
to limit the number of reported tracks based on average OCR confidence scores for a specified language or script.

Please note that the order of the specified language matters. Languages specified first
will have priority (ex. 'eng+deu', English language model will run first and its results will have
priority over German language model).

Example 1: 'eng+deu' = run English, German together as one track detection.
Example 2: 'eng,deu+fra'= run English as the first track and German + French
as the second track.
Example 3: 'fra,script/Latin'= run French as the first track, and Latin script as
the second track.

By default this component contains language model files for Bulgarian (bul),
Chinese - Simplified (chi_sim), German (deu), English (eng), French (fra), Pashto (pus),
Russian (rus), and Spanish (spa) as well as the script model file for Latin (script/Latin).
Note the osd language file (osd.traindata) is for extraction of script orientation rather than language.
Users may download and load in additional language models from https://github.com/tesseract-ocr/tessdata,
stored in the component's `tessdata` directory.

Users may also set Page Segmentation and OCR Engine modes by adjusting TESSERACT_PSM and
TESSERACT_OEM respectively. Please be warned that running TESSERACT_OEM with values 2 or 3 can occasionally
lead to conflicts between the legacy and LSTM engines. Therefore, the default OEM has been set to use the LSTM engine
until this issue is resolved.

The parameter options for the OCR Engine mode are:

TESSERACT_OEM Value| Description
------------- | -------------
0  |  Legacy engine only.
1  |  Neural nets LSTM engine only.
2  |  Legacy + LSTM engines.
3  |  Default, based on what is available.


The parameter options for the Page Segmentation mode are:

TESSERACT_PSM Value| Description
------------- | -------------
0  |  Orientation and script detection (OSD) only.
1  |  Automatic page segmentation with OSD.
2  |  Automatic page segmentation, but no OSD, or OCR.
3  |  Fully automatic page segmentation, but no OSD. (Default)
4  |  Assume a single column of text of variable sizes.
5  |  Assume a single uniform block of vertically aligned text.
6  |  Assume a single uniform block of text.
7  |  Treat the image as a single text line.
8  |  Treat the image as a single word.
9  |  Treat the image as a single word in a circle.
10 |  Treat the image as a single character.
11 |  Sparse text. Find as much text as possible in no particular order.
12 |  Sparse text with OSD.
13 |  Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

For more details please consult the Tesseract command line usage documentation
(https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage).
