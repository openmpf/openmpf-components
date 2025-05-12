# Overview

This repository contains source code for the OpenMPF LLaMa Video Summarization component. The component uses the VideoLLaMA3 model. Refer to these resources:
- Hugging Face: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B
- GitHub: https://github.com/DAMO-NLP-SG/VideoLLaMA3
- White paper: https://arxiv.org/abs/2103.00020

# Job Properties

Refer to `plugin-files/descriptor.json`.

# Track Properties

Returned `VideoTrack` objects have the following members in their `detection_properties`:

| Property Key                     | Description 
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------
| `TEXT`            | Description of activity in the range indicated by the `startOffsetFrame`/`startOffsetTime` and `stopOffsetFrame`/`stopOffsetTime`. When `SEGMENT SUMMARY` is `True`, description summarizes the entire entire segment.
| `SEGMENT ID`      | A unique segment identifier, of the form "<start_offset>-<stop_offset>", where <start_offset> and <stop_offset> are integers indicating the length, in frame counts, of the video (or video segment when a job has been segmented by the Workflow Manager).
| `SEGMENT SUMMARY` | Indicates that `TEXT` summarizes the activity of the entire video (or video segment when a job has been segmented by the Workflow Manager), rather than a single event in a timeline.

# Custom Prompts

For the default prompt refer to `llama_video_summarization_component/default_prompt.txt`.

For the default JSON schema refer to `llama_video_summarization_component/default_schema.json`.

Set `GENERATION_PROMPT_PATH` to specify a file containing a generation prompt to provide the model.
The code requires this to produce JSON that contains all of the fields on the `/default_schema.json`.

Set `GENERATION_JSON_SCHEMA_PATH` to specify a file containing the JSON schema format that
corresponds to the JSON output that the model is instructed to generate. For more information on 
how to specify the schema, refer to these resources: 
- https://json-schema.org/learn/getting-started-step-by-step#create-a-schema-definition
- https://python-jsonschema.readthedocs.io/en/latest/validate/

*[Note: Changes to the JSON schema may require changes to the component code.]*

# Running Component

## Command Line Interface

```bash
docker run --rm -i --gpus '"device=0"' openmpf/openmpf_llama_video_summarization:latest -t video --end 154 -M FRAME_WIDTH=426 -M FRAME_HEIGHT=240 - < tests/data/dog.mp4 > out.json
```