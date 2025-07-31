# Overview

This repository contains source code for the OpenMPF Gemini Detection Component.

This component utilizes a config file that contains any number of prompts for any number of object classes. These prompts and the images/video frames are passed to the Google Gemini server to generate responses.

# Job Properties

The following are the properties that can be specified for the component. All properties except for GEMINI_API_KEY and CLASSIFICATION have default values, making them optional to set.

- `GEMINI_API_KEY`: Your API key to send requests to Google Gemini
- `CLASSIFICATION`: The class of the object(s) in the media. Used to determine the prompt(s). Examples: PERSON and VEHICLE.
- `PROMPT_CONFIGURATION_PATH`: The path to JSON file which contains prompts for specified classifications.
- `JSON_PROMPT_CONFIGURATION_PATH`: The path to a JSON file which contains classes and prompts that specify Gemini to return a JSON object.
- `ENABLE_JSON_PROMPT_FORMAT`: Enables returning a JSON formatted response from Gemini, with the prompt specified at PROMPT_JSON_CONFIGURATION_PATH job property. By default set to false.
- `GENERATE_FRAME_RATE_CAP`: The threshold on the maximum number of frames to process in the video segment within one second of the native video time.
- `MODEL_NAME`: The model to use for Gemini inference. By default it is set to `"gemma-3-27b-it"`.
- `GENERATION_MAX_ATTEMPTS`: The maximum number of times the component will attempt to generate valid JSON output.

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

- Add functionality for generic class property detection
