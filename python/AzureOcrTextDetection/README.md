# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Optical Character Recognition Text Detection Component. This component utilizes 
the [Azure Cognitive Services Optical Character Recognition REST 
endpoint](https://westus.dev.cognitive.microsoft.com/docs/services/56f91f2d778daf23d8ec6739/operations/56f91f2e778daf14a499e1fc)
to extract text from images and videos.

All text extracted from an image can also be tagged using regex and keyword 
tags in a given json file. For keyword tagging, users can provide either words 
or phrases (ex. "bank-tag: \[money, bank of america, etc.]"). Phrases must 
contain words separated by white-space. For more complex pattern matching, 
use regex tags instead. Both forms of tagging are case-insensitive. 

By default, the json tagging file is located in the config folder as 
text-tags.json, however users can provide an alternate full path to a 
tagging file of their choice. English and foreign text tags following 
UTF-8 encoding are supported.

Note that when using `TAGS_BY_REGEX` with multi-word phrases or any regular
expression that includes a space, in many cases you will likely want to use
`"\\s+"` instead of the space character. `"\\s+"` matches a sequence of one or 
more whitespace characters. The recognized text preserves line breaks, so the 
phrase you are looking for may have a line break in the middle of it. 
For example, instead of `"\\bhello world\\b"`, you will likely want to use 
`"\\bhello\\s+world\\b"`. 


# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value. Both can also get the value
from environment variables with the same name. If both are provided, 
the job property will be used. 

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint. 
   e.g. `https://eastus.api.cognitive.microsoft.com/vision/v1.0/ocr`. 
   component has only been tested against v1.0, v2.0, and v2.1 of the API.
   
- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an 
  Azure Cognitive Services account.


# Sample Program
`sample_acs_ocr_text_detector.py` can be used to quickly test with the Azure
endpoint. It only supports images, but the component itself supports both
images and videos.
