# Overview

This repository contains source code for the OpenMPF LLaVA Detection Component.

This component utilizes a config file that contains any number of prompts for any number of object classes. These prompts and the images/video frames are passed to an instance of the [LLaVA 34b](https://huggingface.co/liuhaotian/llava-v1.6-34b) model within [Ollama](https://ollama.com) to generate responses.

The component is built to support multi-stage pipelines where feed forward tracks from an object detector component are passed in to be queried further. The user can specify custom classes and prompts that coincide with the classes of objects being passed into the component.

# Job Properties

The following are the properties that can be specified for the component. Each property has a default value and so none of them necessarily need to be specified for processing jobs.

- `PROMPT_CONFIGURATION_PATH`: Specifies the file path to the prompt config file.
- `JSON_PROMPT_CONFIGURATION_PATH`: Specifies the file path to the JSON output prompt config file.
- `ENABLE_JSON_PROMPT_FORMAT`: Boolean that enables JSON outputs.
- `OLLAMA_SERVER`: The Ollama server `<host>`:`<port>` to use for inferencing.
- `GENERATE_FRAME_RATE_CAP`: Specifies the maximum number of frames to process every second.

# Config File

The config file is a JSON formatted file that is used by the component to know which prompts to ask LLaVA depending on the class of the object. The user can write their own config file and can be used by setting the `PROMPT_CONFIGURATION_PATH` property. The following is an example of the proper syntax to follow:

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

# JSON Config File

The JSON config file allows you to specify that LLaVA returns a JSON object as output where you can specify the fields that the model should fill out. The user can write their own config file and can be used by setting the `JSON_PROMPT_CONFIGURATION_PATH` property. The following is an example of the proper syntax to follow:


```json
{
    "classPrompts": [
        {
            "classes": [
                "DOG"
            ],
            "prompts": [
                "Describe the dog in JSON. The JSON should have the following keys: breed, color, size."
            ]
        },
        {
            "classes": [
                "PERSON"
            ],
            "prompts": [
                "Describe the person in JSON. The JSON should have the following keys: hair_color (if unsure, respond with unsure), clothes, activity."
            ]
        }
    ]
}
```

# Outputs

Once the responses are generated, they are added onto the `detection_properties` dictionary of the associated `ImageLocation` object. for each prompt, the key is specified by the `"detectionProperty"` field of the config JSON and the value will be the LLaVA-generated response.