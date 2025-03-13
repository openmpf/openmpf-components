# Overview

This repository contains source code for the OpenMPF LLaMa Video Summarization component. The component uses the VideoLLaMA3 model. Refer to these resources:
- Hugging Face: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B
- GitHub: https://github.com/DAMO-NLP-SG/VideoLLaMA3
- White paper: https://arxiv.org/abs/2103.00020

# Job Properties

Refer to `plugin-files/descriptor.json`.

## Detection Properties

Returned `VideoTrack` objects have the following members in their `detection_properties`:

| Property Key                     | Description 
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------
| `TEXT`                 | General summary of the entire video.
| `VIDEO LENGTH`         | Length of the video in seconds. Used to determine if the `VIDEO EVENT TIMELINE` covers the full length.
| `VIDEO EVENT TIMELINE` | JSON representation of the video event timeline.

# Custom Prompts

For the default prompt refer to `llama_video_summarization_component/default_prompt.txt`.

For the default JSON schema refer to `llama_video_summarization_component/default_schema.json`.

Set `GENERATION_PROMPT_PATH` to specify a file containing a generation prompt to provide the model.
Currently, the code requires this to produce JSON that containts the following fields:
- `video_summary`
- `video_length`
- `video_event_timeline`
    - `video_event_timeline.timestamp_end`

Set `GENERATION_JSON_SCHEMA_PATH` to specify a file containing the JSON schema format that
corresponds to the JSON output that the model is instructed to generate. For more information on 
how to specify the schema, refer to these resources: 
- https://json-schema.org/learn/getting-started-step-by-step#create-a-schema-definition
- https://python-jsonschema.readthedocs.io/en/latest/validate/