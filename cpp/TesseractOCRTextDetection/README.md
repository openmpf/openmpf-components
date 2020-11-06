# Overview

This repository contains source code and model data for the OpenMPF Tesseract
OCR text detection component.

The component extracts text found in an image, reported as a single track
detection. PDF documents can also be processed with one track detection per
page. The first page corresponds to the detection property `PAGE_NUM=1`. For
debugging purposes, images converted from documents are stored in a temporary
job directory under `plugin/TesseractOCR/tmp-[job-id]-[random tag]`. This
directory is removed when the job completes successfully.

Please refer to https://imagemagick.org/script/formats.php for support of other
document file formats.

Users may set the language of each track using the `TESSERACT_LANGUAGE` parameter
as well as adjust image preprocessing settings for text extraction.

# Tessdata Models

Language models supported by Tesseract are stored by default in
`$MPF_HOME/plugins/TesseractOCRTextDetection/tessdata` directory and script
models are stored in `tessdata/script`. Users can set a new tessdata directory
by modifying the `MODELS_DIR_PATH` job property. Once set, the component will look
for tessdata files in `[MODELS_DIR_PATH]/TesseractOCRTextDetection/tessdata`.

By default, the component will first check for models in the `MODELS_DIR_PATH`
followed by the default tessdata path. Please ensure that any language models
that run together are stored together in the same directory (i.e. while running
`eng+bul`, both `eng.traineddata` and `bul.traineddata` should be stored
together in at least one specified directory, while running `eng,bul` the models
can be stored separately).

Additional tessdata models can then be added to the specified tessdata folder to
expand supported languages and scripts.

Each language module follows ISO 639-2 designations, with character variations
of a language (ex. `uzb_cyrl`) also supported. Users must enter the same
designation to enable the corresponding language detection (ex.
`TESSERACT_LANGUAGE=deu` for German text). Users will be warned when a given
language is not supported when the corresponding language module cannot be
located in this directory.

Each script module is contained within the `tessdata/script` directory with the
first letter of its type capitalized (ex. `tessdata/script/Latin.traineddata`).
Users will need to specify the `script/` path followed by the full name of the
script being processed. (ex. `TESSERACT_LANGUAGE=script/Latin` will enable Latin
script text extraction).

# Detecting Multiple Languages

There are two options to run multiple user-specified languages/scripts. Users can separate each
specified language and script using the `+` delimiter to run multiple models
together in one track and `,` to run them as separate tracks. Delimiters
can also be combined for separate multilingual tracks. Lastly, users can set `MAX_TEXT_TRACKS` to limit the number of
reported OCR tracks so that only the tracks with the highest scores are reported.
Note that if this is set lower than the number of scripts detected by OSD, then the OCR tracks for the scripts with the
lowest scores will not be reported.

Please note that the order of the specified language matters. Languages specified first
will have priority (ex. `eng+deu`, English language model will run first and its results will have
priority over German language model). When specifying multiple scripts using `,` or `+`, duplicate or redundant
script requests are ignored.

Example 1: `eng+deu` means run English, German together as one track detection.

Example 2: `eng,deu+fra` means run English as the first track and German + French
as the second track.

Example 3: `fra,script/Latin` means run French as the first track, and Latin script as
the second track.

Example 4: `eng,eng+bul,eng+eng` will only return two tracks, English only as first track
followed by English+Bulgarian as the second track.

By default this component contains language model files for:
* Bulgarian (`bul`)
* Chinese - Simplified (`chi_sim`)
* German (`deu`)
* English (`eng`)
* French (`fra`)
* Pashto (`pus`)
* Russian (`rus`)
* Spanish (`spa`)

As well as the script model file for:
* Latin (`script/Latin`)

Note the OSD language file (`osd.traindata`) is for extraction of script orientation rather than language.
Users may download additional language/script models from https://github.com/tesseract-ocr/tessdata and place them in the component's `tessdata` directory or `[MODELS_DIR_PATH]/TesseractOCRTextDetection/tessdata`.
During processing, if the OSD model detects a certain language but the corresponding language model is missing from the component's `tessdata` directory, then that language will be reported in the `MISSING_LANGUAGE_MODELS` output parameter.
When all OSD-detected language models are missing in the `tessdata` directory, the component will default to running the `TESSERACT_LANGUAGE` model instead.
Please note that the job will fail instead if any models specified in `TESSERACT_LANGUAGE` are missing.

# OSD Automation

Users can set `ENABLE_OSD_AUTOMATION` to true to enable automatic orientation and script detection:
* An additional 8 parameters are reported in corresponding OCR tracks:
    * `OSD_PRIMARY_SCRIPT`, `OSD_SECONDARY_SCRIPTS`, and `ROTATION` of text (0, 90, 180, and 270 degrees counterclockwise) in an image.
    * `ROTATION` represents the current counterclockwise orientation of the text. Thus when `ROTATION=90`, Tesseract will apply a 90 degree clockwise rotation to reverse the text to an upright position.
    * For primary and secondary detected scripts, raw scores for each prediction are stored inside of `OSD_PRIMARY_SCRIPT_SCORE` and `OSD_SECONDARY_SCRIPT_SCORES`.
    * For the primary script, `OSD_PRIMARY_SCRIPT_CONFIDENCE` is also generated by Tesseract, specifying the relative confidence of the primary script score against the second top-detected script score.
    * Similarly, `OSD_TEXT_ORIENTATION_CONFIDENCE` represents the relative confidence of the top text orientation prediction score against the second-best text orientation prediction score. Raw text orientation scores are excluded from the report as the individual values are not normalized (large non-positive values that provide little insight into prediction confidence, unlike individual script scores).
    * `OSD_FALLBACK_OCCURRED` notifies users if a second round of OSD was performed (see `ENABLE_OSD_FALLBACK` below).
* If the detected text orientation is >= `MIN_OSD_TEXT_ORIENTATION_CONFIDENCE` threshold, then the frame will automatically be rotated 0, 90, 180, or 270 degrees before performing OCR. If the threshold is not exceeded, then OCR is performed on the default orientation (0 degree rotation).
* If the detected primary script confidence is >= the `MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE` threshold, and script score is >= the `MIN_OSD_SCRIPT_SCORE` threshold, then OCR will be performed for the script and the `TESSERACT_LANGUAGE` setting will be ignored. If either threshold is not exceeded, then OCR is performed using the `TESSERACT_LANGUAGE` setting.
* If OSD detects multiple scripts, and `MAX_OSD_SCRIPTS` is >= 2, then OCR will be performed on each detected script given these rules:
    * Each detected script must be >= the `MIN_OSD_SCRIPT_SCORE` threshold.
    * The detected script with the highest score is considered the primary script. The others are considered secondary scripts. Secondary scripts must have scores are >= the `MIN_OSD_SECONDARY_SCRIPT_THRESHOLD`, which is a % applied primary script score. For example, if `MIN_OSD_SECONDARY_SCRIPT_THRESHOLD=0.8`, then the secondary scripts must have scores that are at least 80% of the primary script score.
    * Note that if the number of detected scripts exceeds the `MAX_OSD_SCRIPTS` setting, then only the scripts with the highest scores are considered.
    * If `COMBINE_OSD_SCRIPTS` is set to true (default setting), a single output track will be generated using a combination of all detected scripts (primary and secondary) that are considered acceptable for processing. Otherwise one track is generated for each accepted script.
* When `ENABLE_OSD_FALLBACK` is enabled, an additional round of OSD is performed when the first round fails to generate script predictions that are above the OSD score and confidence thresholds.
    * In the second pass, the component will run OSD on multiple copies of the input text image to get an improved prediction score.
    * The results are then kept depending on the minimum OSD script thresholds.
    * The `OSD_FALLBACK_OCCURRED` parameter will be set to true if the second pass is performed.

* When using one of the PSM modes that performs OSD and OCR, if the detected script with the highest confidence is `Common` (a.k.a. the numeric script model), then Tesseract will choose the script with the second-highest confidence to perform OCR. For consistency, this component performs the same behavior when `ENABLE_OSD_AUTOMATION` is true. Thus, when `Common` becomes the primary script, the component will instead search for the next best secondary script for OCR text processing, or use the `TESSERACT_LANGUAGE` setting when the secondary script score is too low.

# Two-pass OCR

If `ROTATE_AND_DETECT` is enabled, the component will perform two-pass OCR on images as follows:
* The first pass will use the rotation detected by OSD if enabled, otherwise it will run on the image directly.
* The second pass will apply another 180 degree rotation on the image before processing.
    * Users can set `ROTATE_AND_DETECT_MIN_OCR_CONFIDENCE` to a positive value to enable confidence checks on the first OCR pass.
    * If the first pass is greater than or equal to than the minimum specified threshold, the second pass of the OCR is skipped.
* After performing two-pass OCR, the output with the highest OCR confidence is kept as the final track.
* For multiple, separate language tracks, the previous steps are repeated for each specified language track, such that one high confidence output for each track is returned by the component.
* The component will report `ROTATE_AND_DETECT_PASS` alongside the `ROTATION` property. The `ROTATION` property is the rotation of the text region, taking into consideration both the OSD rotation (if OSD was performed), as well as the 2-pass rotation attempt. `ROTATE_AND_DETECT_PASS` is set to 0 or 180 depending on which pass produced the best results.

# Page Segmentation and OCR Engine Modes

Users may also set Page Segmentation and OCR Engine modes by adjusting `TESSERACT_PSM` and
`TESSERACT_OEM`, respectively. Please be warned that running `TESSERACT_OEM` with values 2 or 3 can occasionally
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

# Parallel OCR model and PDF processing:

Users can set the `MAX_PARALLEL_SCRIPT_THREADS` and `MAX_PARALLEL_PAGE_THREADS` to enable and adjust
parallel processing behavior in this component.

For image processing only, `MAX_PARALLEL_SCRIPT_THREADS` limits the maximum number of active threads, with each thread running one pass of OCR on an image.
For document processing only, `MAX_PARALLEL_PAGE_THREADS` limits the maximum number of active threads, with each thread processing OCRs serially on a single page or image from that document.

When `MAX_PARALLEL_SCRIPT_THREADS` or `MAX_PARALLEL_PAGE_THREADS` is set to a value of 1 or less, parallel threading of multiple OCRs or pages is disabled respectively.
