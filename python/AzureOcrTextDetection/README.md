# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Optical Character Recognition Text Detection Component. This component utilizes 
the [Azure Cognitive Services Optical Character Recognition REST 
endpoint](https://westus.dev.cognitive.microsoft.com/docs/services/56f91f2d778daf23d8ec6739/operations/56f91f2e778daf14a499e1fc)
to extract text from images and videos.


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
