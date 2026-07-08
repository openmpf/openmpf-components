# Gemini Video Summarization Component

This component creates text summaries, and optionally event timelines, for OpenMPF video jobs. The component preprocesses each video segment, sends the prepared media and prompt to a model backend, validates that the model returned JSON, and emits OpenMPF `TEXT` tracks.

## Job Properties Used By The Component

These are the properties read directly by the current component code.

| Property | Default | Description |
| :--- | :--- | :--- |
| `API` | `OpenAI` | Backend API to use for model inference. Supported values are `OpenAI` and `Google`. |
| `OPENAI_BASE_URL` | empty | Optional base URL for an OpenAI-compatible API service such as vLLM. Leave empty to use the OpenAI client default. |
| `MODEL_NAME` | empty | Model name sent to the backend. Required for API calls. |
| `APPLICATION_CREDENTIALS` | empty | Path to a credential file or the name of an environment variable containing credentials. |
| `GENERATION_PROMPT_PATH` | `data/default_prompt.txt` or `data/default_prompt_no_tl.txt` | Optional path to a prompt file. If unset, the component selects the timeline or no-timeline default based on `ENABLE_TIMELINE`. |
| `ENABLE_TIMELINE` | `1` | `1` asks the model for a summary and event timeline. `0` asks only for a summary. |
| `GENERATION_MAX_ATTEMPTS` | `5` | Number of attempts for getting valid JSON and, when enabled, a valid timeline. |
| `TIMELINE_CHECK_TARGET_THRESHOLD` | `10` | Number of seconds timeline events may fall outside segment bounds before retrying. Set to `-1` to disable this check. |
| `PROCESS_FPS` | `1.0` | Target FPS for the video supplied to the model. The component never exceeds the source FPS. If this matches the source FPS, the component can pass the source MP4 directly or remux without reencoding. |
| `PROJECT_ID` | empty | Google Cloud project id. Used by the Google/Vertex path. |
| `BUCKET_NAME` | empty | GCS bucket used by the Google/Vertex path for temporary model input video upload. |
| `LABEL_PREFIX` | empty | Optional prefix for Vertex AI request labels. |
| `LABEL_USER` | empty | Optional user label value. |
| `LABEL_PURPOSE` | empty | Optional purpose label value. |
| `ENABLE_AUDIO` | `1` | For OpenAI-compatible requests, extract the original segment audio and append it as an `audio_url` when the model name is recognized as audio-capable. Currently recognized audio-capable names include `gemma-4-e2b`, `gemma-4-e4b`, and `gemma-4-12b`. |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `600` | Timeout for OpenAI-compatible chat completion requests. |
| `OPENAI_MAX_RETRIES` | `0` with `base_url`, otherwise `2` | Retry count for the OpenAI client. vLLM runs normally use `0` because the tester/component controls retry behavior separately. |
| `OPENAI_MAX_TOKENS` | empty | Optional `max_tokens` value for OpenAI-compatible chat completions. Empty means server default. |
| `OPENAI_TEMPERATURE` | empty | Optional temperature for OpenAI-compatible chat completions. Empty means server default. |
| `OPENAI_RESPONSE_FORMAT_JSON_OBJECT` | `false` with `base_url`, otherwise `true` | Sends `response_format={"type":"json_object"}` when true. This is disabled by default for vLLM because guided JSON can fail with some vLLM/model combinations. |
| `KEEP_TEMP_MEDIA` | `0` | When true, keeps temporary preprocessed MP4 and extracted WAV files for debugging. |

## Motion Profiling Properties

Motion profiling no longer duplicates frames. Instead, it finds high-motion regions and adds a text hint to the prompt telling the model to reconsider those time ranges after reviewing the full video/audio.

| Property | Default | Description |
| :--- | :--- | :--- |
| `MOTION_EMPHASIS_SCORE_FPS` | `5.0` | FPS used for motion scoring. The effective score FPS is at least `PROCESS_FPS` and at most the source FPS. Empty means use source FPS. |
| `MOTION_EMPHASIS_SCORE_WIDTH` | `320` | Width used for motion-score frames. Scoring frames are downscaled only for motion calculation. The supplied model video remains full resolution. Set to `0` to score at original width. |
| `MOTION_EMPHASIS_THRESHOLD` | `5.0` | Motion score threshold. Set to an empty value to use `MOTION_EMPHASIS_PERCENTILE` instead. |
| `MOTION_EMPHASIS_PERCENTILE` | `90.0` | Percentile used to choose the motion threshold when `MOTION_EMPHASIS_THRESHOLD` is empty. |
| `MOTION_EMPHASIS_NEIGHBOR_FRAMES` | `0` | Adds time padding around high-motion points when building motion-focus ranges. This does not duplicate frames. |
| `MOTION_FOCUS_MAX_RANGES` | `40` | Maximum number of motion-focus ranges included in the prompt. Set to `0` to disable motion-focus hints. |

These are OpenMPF pipeline/segmenter properties rather than properties consumed directly by this Python component:

- `TARGET_SEGMENT_LENGTH`
- `VFR_TARGET_SEGMENT_LENGTH`
- `SEGMENT_LENGTH_SPECIFICATION`
- `MERGE_TRACKS`
- `QUALITY_SELECTION_PROPERTY`

They may still be useful in the OpenMPF pipeline, but they are not read from `job.job_properties` by the component implementation.

## Custom Prompts

The default timeline prompt is:

- `gemini_video_summarization_component/data/default_prompt.txt`

The default no-timeline prompt is:

- `gemini_video_summarization_component/data/default_prompt_no_tl.txt`

Set `GENERATION_PROMPT_PATH` to use a mounted custom prompt file. If timelines are enabled, keep the JSON output instructions and timestamp-format instructions in the prompt. The component expects MM:SS timestamps and remaps timestamps when it sends a sampled/preprocessed video to the model.

## Audio Behavior

When `ENABLE_AUDIO=1`, the OpenAI-compatible path extracts the original audio for the same video segment with ffmpeg/imageio-ffmpeg. The extracted audio is sent as a separate WAV data URL after the text prompt. The generated audio is mono, 16 kHz, float32 PCM WAV.

Audio is only sent when the model name is recognized as audio-capable. The current code recognizes `gemma-4-e2b`, `gemma-4-e4b`, and `gemma-4-12b`. For example, `gemma-4-31b` is not treated as audio-capable.

## Docker Container

Example OpenMPF Compose service for the OpenAI-compatible API path:

```yaml
gemini-video-summarization:
  <<: *detection-component-base
  image: <IMAGE>
  volumes:
    - host_directory/openai_api_key.txt:/run/secrets/openai_api_key.txt:ro
    - host_directory/prompt_file.txt:/opt/mpf/share/prompt_file.txt:ro # optional
    - shared_data:/opt/mpf/share
  environment:
    - MPF_PROP_API=OpenAI
    - MPF_PROP_OPENAI_BASE_URL=http://vllm:8000/v1
    - MPF_PROP_MODEL_NAME=<MODEL NAME>
    - MPF_PROP_APPLICATION_CREDENTIALS=/run/secrets/openai_api_key.txt
    - MPF_PROP_GENERATION_PROMPT_PATH=/opt/mpf/share/prompt_file.txt # optional
    - MPF_PROP_ENABLE_TIMELINE=0
    - MPF_PROP_ENABLE_AUDIO=1
    - MPF_PROP_PROCESS_FPS=1.0
    - MPF_PROP_OPENAI_REQUEST_TIMEOUT_SECONDS=600
```

`MODEL_NAME` should match the model exposed by the OpenAI-compatible service. For a local vLLM server, run vLLM separately and configure this component deployment to use that service as the OpenAI base URL.
