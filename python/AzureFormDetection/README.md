# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Form Detection Component. This component utilizes the [Azure Cognitive Services Form Detection REST
endpoint](https://westus2.dev.cognitive.microsoft.com/docs/services/form-recognizer-api-v2/operations/AnalyzeLayoutAsync)
to extract formatted text from documents (PDFs) and images.


# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value. Both can also get the value
from environment variables with the same name. If both are provided, 
the job property will be used. 

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint. Multiple model processing options exist for form detection
   and can be specified through the URL as follows:

   - `https://{endpoint}/formrecognizer/v2.0/layout/analyze` - v2.0 layout analysis.
   - `https://{endpoint}/formrecognizer/v2.1-preview.1/Layout/analyze` - v2.1 layout analysis.
   - `https://{endpoint}/formrecognizer/v2.1-preview.1/prebuilt/businessCard/analyze` - v2.1 business card model.
   - `https://{endpoint}/formrecognizer/v2.1-preview.1/prebuilt/receipt/analyze` - v2.1 receipt model.
   - `https://{endpoint}/formrecognizer/v2.1-preview.1/custom/models/{modelId}/analyze` - v2.1 custom model.

   This component has only been tested against v2.0 and v2.1 of the API.
   
- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an 
  Azure Cognitive Services account.

Optional job properties include:
- `MERGE_LINES`: A boolean toggle to specify whether text detection in a page should appear as separate `LINE` outputs or
   a single `MERGED_LINES` output. The default is to merge lines as an excessive number of detections may be reported
   otherwise.
- `INCLUDE_TEXT_DETAILS`: A boolean toggle to specify whether text output should also be provided alongside tables
   and key-value pairings produced by form detection models. Set to true by default.
- `LANGUAGE`:  When provided, disables automatic language detection and processes document based on provided language.
   The language code must be in case-sensitive BCP-47.  Currently, only English (`en`), Dutch (`nl`),
   French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), simplified Chinese (`zh-Hans`) and Spanish (`es`)
   are supported. English is the only language that supports handwritten characters in addition to printed characters.

# Job Outputs
Currently the component provides line-based outputs, key-value pairs, csv table outputs, and custom forms in separate detection tracks.

- Detection tracks with `OUTPUT_TYPE` set to `LINE` or `MERGED_LINES` will include `TEXT` results. `MERGED_LINES`
  contains `TEXT` output where each line is separated by a newline character.

- Detection tracks with `OUTPUT_TYPE` set to `TABLE` will contain `TABLE_CSV_OUTPUT`, a CSV formatted table,
  instead of `TEXT`.

- Detection tracks with `OUTPUT_TYPE` set to `KEY_VALUE_PAIRS` will contain `KEY_VALUE_PAIRS_JSON` instead of `TEXT`.
  Each key-value pair in `KEY_VALUE_PAIRS_JSON` follows the JSON object format:
  `{"<key_1>": "<val_1>", "<key_2>": "<val_2>", ...}`

  Please note that `KEY_VALUE_PAIRS` are produced by models trained on unlabeled documents.
  For more information on custom model outputs, please see [below](#custom-models).

- Detection tracks with `OUTPUT_TYPE` set to `DOCUMENT_RESULT` are produced by pretrained receipt/business card models
  as well as custom models trained on labeled data. For more information on custom model outputs, please see [below](#custom-models).

Results are organized by `PAGE_NUM` and output indexes (`LINE_NUM`, `TABLE_NUM`, `DOCUMENT_RESULT_NUM`) with 1 as the starting index.
For images, the bounding box information for lines, merged lines, whole tables, and custom forms is also provided in the detection
track.

## Custom Models
ACS form recognizer also supports training and deployment of custom form detection models, as well as providing
pretrained models for detecting business cards and receipts.

Two types of custom model outputs exist, depending on whether the custom model was trained on unlabeled or labeled form data.

- Unlabeled data training example: https://docs.microsoft.com/en-us/azure/cognitive-services/form-recognizer/quickstarts/python-train-extract?tabs=v2-0

- Labeled data training example: https://docs.microsoft.com/en-us/azure/cognitive-services/form-recognizer/quickstarts/python-labeled-data?tabs=v2-0

For custom models trained without labeled data, additional results will be provided and stored as `KEY_VALUE_PAIRS`
output tracks, with one detection track per page.

For custom models trained using labeled data, additional custom form output is stored in
separate detection tracks with the output type set to `DOCUMENT_RESULT`. The pretrained models for business cards and receipts
were also trained on labeled data and will generate `DOCUMENT_RESULT` tracks.

For labeled `DOCUMENT_RESULT` tracks, each output track stores a separate document form, receipt, or business card detected in the image or pdf and has the following
unique detection properties:

-  `DOCUMENT_TYPE`: Type of document form detected in the current track (receipt, business card, etc.).
-  `PAGE_RANGE` : Species the page range for the detected custom form. Currently, business cards
    and receipts must be restricted to single-pages (i.e. each receipt or card must appear in one page, not across
    multiple pages).
-  `DOCUMENT_RESULT_NUM`: Each document form track will be organized based on its index from the ACS `documentResults` output, starting from 1.
-  `DOCUMENT_JSON_FIELDS`: Contains a JSON-formatted object of various custom form fields detected in the document. Details on custom form,
    business card, and receipt fields can be found at https://westcentralus.dev.cognitive.microsoft.com/docs/services/form-recognizer-api-v2-1-preview-1/operations/GetAnalyzeFormResult.

# Sample Program
`sample_acs_form_detector.py` can be used to quickly test with the Azure
endpoint. It supports images and PDF inputs only. PDF inputs must contain the `.pdf`
extension to be properly recognized as generic job inputs. Otherwise, the sample code
will default to running an image job on the provided input.
