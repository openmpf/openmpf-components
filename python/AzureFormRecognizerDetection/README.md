# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Form Recognition Text Detection Component. This component utilizes
the [Azure Cognitive Services Form Recognizer REST
endpoint](https://westus2.dev.cognitive.microsoft.com/docs/services/form-recognizer-api-v2/operations/AnalyzeLayoutAsync)
to extract formatted text from documents (PDFs) and images.


# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value. Both can also get the value
from environment variables with the same name. If both are provided, 
the job property will be used. 

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint. 
   e.g. `https://{endpoint}/formrecognizer/v2.0/layout/analyze`.
   component has only been tested against v2.0 of the API.
   
- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an 
  Azure Cognitive Services account.

Optional job properties include:
- 'MERGE_LINES': A boolean toggle to specify whether text detection in a page should appear as separate 'LINE' outputs or
   a single 'MERGED_LINES' output. The default is to merge lines as an excessive number of detections may be reported
   otherwise.
-  'INCLUDE_TEXT_DETAILS': A boolean toggle to specify whether text output should also be provided alongside tables
   and key-value pairings produced by form recognizer models. Set to true by default.

# Job Outputs
Currently the component provides line-based outputs and csv table outputs in separate detection tracks.
Detection tracks with 'OUTPUT_TYPE' set to 'LINE' or 'MERGED_LINES' will include 'TEXT' results, whereas tracks
with 'OUTPUT_TYPE' set to 'TABLE' will contain a single 'TABLE_CSV_OUTPUT'.

Results are organized by 'PAGE_NUM' and output indexes ('LINE_NUM', 'TABLE_NUM') with 0 as the starting index.
For images, the bounding box information (for lines, merged lines, and whole tables) is also provided in the detection
track.


# Sample Program
`sample_acs_form_recognizer_detector.py` can be used to quickly test with the Azure
endpoint. It supports images and PDF inputs only. PDF inputs must contain the `.pdf`
extension to be properly recognized as generic job inputs. Otherwise, the sample code
will default to running an image job on the provided input.
