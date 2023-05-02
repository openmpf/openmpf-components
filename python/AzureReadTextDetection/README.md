# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Read Text Detection Component, which utilizes the [Azure Cognitive Services Read Detection REST
endpoint](https://westus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/5d986960601faab4bf452005)
to extract formatted text from documents (PDFs), images, and videos.

**Moving forward, future Read OCR enhancements will fall under the following two services:**
- [Computer Vision Version 4.0 Read OCR](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-ocr) (in preview).
   - For processing general, non-document images containing text (text in the wild).
- [Form Recognizer Read Model v3.0](https://learn.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/concept-read?view=form-recog-3.0.0) (in preview).
   - For processing scans of documents and clean text.

We are currently reviewing and moving to support Azure's v4.0 Read OCR for processing images containing text.

# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value. Both can also get the value
from environment variables with the same name. If both are provided,
the job property will be used.

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint. Multiple Read API versions exist for text detection
   and can be specified through the URL as follows:

   - `https://{endpoint}/vision/v3.0/read/analyze`            - v3.0 Read OCR analysis.
   - `https://{endpoint}/vision/v3.1/read/analyze`            - v3.1 Read OCR analysis.
   - **`https://{endpoint}/vision/v3.2/read/analyze`            - v3.2 GA Read OCR analysis. (Final version for version 3 Read OCR)**

   Note that version 3.2 supports a greater number of auto-detected OCR languages, including handwritten text (see [v3.2 supported languages](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/language-support#optical-character-recognition-ocr)).

   Currently listed handwritten language support includes English, Chinese, French, German, Italian, Japanese, Korean, Portuguese, and Spanish.

- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an
  Azure Cognitive Services account.

Optional job properties include:
- `MERGE_LINES`: A boolean toggle to specify whether text detection in a page should appear as separate `LINE` outputs or
   a single `MERGED_LINES` output. The default is to merge lines as an excessive number of detections may be reported
   otherwise.
- `LANGUAGE`:  When provided, disables automatic language detection and processes document based on provided language.
   The language code must be in case-sensitive BCP-47.  Currently, only English (`en`), Dutch (`nl`),
   French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), simplified Chinese (`zh-Hans`) and Spanish (`es`)
   are supported. English is the only language that supports handwritten characters in addition to printed characters.

# Job Outputs
Currently the component provides text outputs in separate detection tracks.

Detection tracks with `OUTPUT_TYPE` set to `LINE` or `MERGED_LINES` will include `TEXT` results. `MERGED_LINES`
contains `TEXT` output where each line is separated by a newline character.
Confidence scores are calculated as the average confidence of individual words detected in a single `LINE`
or across all `MERGED_LINES`.

Results are organized by `PAGE_NUM` and output indexes (`LINE_NUM`) with 1 as the starting index.
For images, the bounding box information for lines or merged lines is also provided in the detection track.

# Sample Program
`sample_acs_read_detector.py` can be used to quickly test with the Azure
endpoint. It supports images, video (`*.avi`), and PDF inputs only. PDF inputs must contain the `.pdf`
extension to be properly recognized as generic job inputs. Similarly video inputs must contain the ".avi" extension.
Otherwise, the sample code will default to running an image job on the provided input.
