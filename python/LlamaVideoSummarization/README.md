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

# Behavior

VideoLLaMA3 runs in one of two user settable modes: 'fps' (frames per second) or 'uniform'. OpenMPF attempts to run the model in 'fps' mode. However, VideoLLaMA3 runs an internal calculation* of the video duration divided by the video fps, and checks to see if it is less than max_frames. If it exceeds max_frames, the model runs in 'uniform' mode, which uses a uniform sampling of frames from the provided video. For very long videos, this can result in a scenario where the sample rate is much lower than 1 FPS. For example, a 30 minute video might be sampled with an effective sample rate of 0.1 fps, or one frame every 10 seconds.

Additionally, VideoLLaMA3 appears to be optimized to process videos with a duration of 3 minutes or less. Due to this and the considerations outlined above, the default behavior of this component within the OpenMPF Workflow Manager (WFM) is to divide videos into 3 minute segments for processing.

\* See https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/213157c28fe5b52aa8c235c60965c16db69b9c7d/videollama3/mm_utils.py#L327