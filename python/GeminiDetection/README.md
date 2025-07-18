# Overview

This repository contains source code for the OpenMPF Gemini Detection Component.

This component utilizes a config file that contains any number of prompts for any number of object classes. These prompts and the images/video frames are passed to the Google Gemini server to generate responses.

# Job Properties

The following are the properties that can be specified for the component. Each property has a default value and so none of them necessarily need to be specified for processing jobs.

- `PROMPT_CONFIGURATION_PATH`: Path to JSON file which contains prompts for specified classifications.
- `GEMINI_API_KEY`: Your API key to send requests to Google Gemini

# Config File

The config file is a JSON formatted file that is used by the component to know which prompts to ask Gemini depending on the class of the object. The user can write their own config file and can be used by setting the `PROMPT_CONFIGURATION_PATH` property. The following is an example of the proper syntax to follow:

```json
[
    {
        "classes": [
            "DOG",
            "CAT",
            "HORSE"
        ],
        "prompts": [
            {
                "detectionProperty": "DESCRIPTION",
                "prompt": "Describe the animal's color and appearance."
            }
        ]
    },
    {
        "classes": [
            "DOG"
        ],
        "prompts": [
            {
                "detectionProperty": "DOG BREED",
                "prompt": "Describe the potential breeds that this dog could contain."
            }
        ]
    }
]
```

Note that a class can appear in multiple entries in the JSON, such as `"DOG"` in the example. If you have multiple classes that share a prompt, you can list them together like above and then add more questions for each individual class if you wish to get more specific. 

Also be sure to make each `"detectionProperty"` distinct for a given class so that none of your prompts are overwritten.

# Outputs

Once the responses are generated, they are added onto the `detection_properties` dictionary of the associated `ImageLocation` object. for each prompt, the key is specified by the `"detectionProperty"` field of the config JSON and the value will be the Gemini-generated response.

# TODO

- Implement feed forward jobs
- Implement JSON response jobs
- Fix mocking unittests so API key isn't required to test component
