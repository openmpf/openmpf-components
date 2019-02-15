# Overview

This repository contains source code and model data for the OpenMPF Tesseract
OCR text detection component.

The component extracts text found in an image, reported as a single track detection.
PDF documents can also be processed with one track detection per page. The first page
corresponds to the detection property PAGE_NUM = 1. For debugging purposes, images converted
from documents are stored in a temporary job directory under
plugin/TesseractOCR/tmp-[job-id]-[random tag]. This directory is removed when the job completes successfully.

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


Language models supported by Tesseract must be stored in /bin/tessdata.
Users are able to store new models into the tessdata folder to expand
supported languages.

Most languages will only require the *.traineddata file. Certain languages will
also require the [lang].cube.* files to be present within the tessdata directory as well.

Each language module follows ISO 639-2 designations, with character variations
of a language (ex. uzb_cyrl) also supported. Users must enter the same designation
to enable the corresponding language detection (ex. TESSERACT_LANGUAGE = "deu"
for German text). Users will be warned when a given language is not supported when the
corresponding language module cannot be located in this directory.

There are two options to run multiple languages. Users can separate each
specified language using the '+' delimiter to run multiple languages
together in one track and ',' to run them as separate tracks. Delimiters
can also be combined for separate multilingual tracks.

Example 1: 'eng+deu' = run English, German together as one track detection.
Example 2: 'eng, deu+fra'= run English as the first track and German + French
as the second track. Languages that use a .cube model file should be specified
last to avoid Tesseract language model errors (ex. cube_lang+eng will trigger errors
while eng+cube_lang will work properly).


By default this component contains model files for Bulgarian (bul),
Chinese - Simplified (chi_sim), German (deu), English (eng), French (fra), Pashto (pus),
Russian (rus), and Spanish (spa). Note the osd language file (osd.traindata) is
for extraction of script orientation rather than language. Users may download and
load in additional language models from https://github.com/tesseract-ocr/tessdata/tree/3.04.00,
stored in the component's bin/tessdata directory.

Users may also set Page Segmentation and OCR Engine modes by adjusting TESSERACT_PSM and
TESSERACT_OEM respectively.

The parameter options for the OCR Engine mode are:

TESSERACT_OEM Parameters| Description
------------- | -------------
0  |  Legacy engine only.
1  |  Neural nets LSTM engine only.
2  |  Legacy + LSTM engines.
3  |  Default, based on what is available.


The parameter options for the Page Segmentation mode are:

TESSERACT_PSM Parameters| Description
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
