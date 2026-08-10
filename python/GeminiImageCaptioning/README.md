# Gemini Image Captioning Component

This component sends images and sampled video frames with configurable prompts to a multimodal model. It supports OpenAI-compatible chat completion services, including vLLM, and Google Vertex AI. OpenAI is the default backend.

# Job Properties

| Property | Default | Description |
| --- | --- | --- |
| `API` | `OpenAI` | Backend to use. Supported values are `OpenAI` and `Google`. |
| `OPENAI_BASE_URL` | empty | Optional base URL for an OpenAI-compatible service such as vLLM. Leave empty to use the OpenAI client default. |
| `APPLICATION_CREDENTIALS` | empty | For OpenAI, a file containing the API key or the name of an environment variable containing it. For Google, the path to a Google application credential file. |
| `PROJECT_ID` | empty | Google Cloud project id for the Google backend. |
| `LABEL_PREFIX` | empty | Optional prefix for Google request labels. |
| `LABEL_USER` | empty | Optional user value for Google request labels. |
| `LABEL_PURPOSE` | empty | Optional purpose value for Google request labels. |
| `CLASSIFICATION` | empty | Object class used to select prompts, such as `PERSON` or `VEHICLE`. |
| `PROMPT_CONFIGURATION_PATH` | bundled file | Path to a JSON file containing prompts for classifications and full frames. |
| `JSON_PROMPT_CONFIGURATION_PATH` | bundled file | Path to a JSON file containing prompts that request JSON responses. |
| `ENABLE_JSON_PROMPT_FORMAT` | `false` | Use the JSON prompt configuration and parse the response into detection properties. |
| `GENERATE_FRAME_RATE_CAP` | `1.0` | Maximum number of frames processed per second of native video. Values less than or equal to zero process every frame. |
| `MODEL_NAME` | `google/gemma-4-12B-it` | Model exposed by the selected backend. |
| `GENERATION_MAX_ATTEMPTS` | `5` | Maximum attempts to obtain valid JSON output. |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `600` | Timeout for OpenAI-compatible chat completion requests. |
| `OPENAI_MAX_RETRIES` | `0` with a base URL, otherwise `2` | Retry count used by the OpenAI client. |
| `OPENAI_MAX_TOKENS` | empty | Optional `max_tokens` request value. |
| `OPENAI_TEMPERATURE` | empty | Optional temperature request value. |
| `OPENAI_RESPONSE_FORMAT_JSON_OBJECT` | `false` with a base URL, otherwise `true` | Request JSON object output when JSON prompt mode is enabled. |

When `APPLICATION_CREDENTIALS` is empty or invalid for an OpenAI-compatible service, the client uses a dummy key. This supports local services that do not require authentication. Google jobs require an existing credential file.

# Prompt Configuration

The prompt configuration maps one or more classes to detection-property names and prompts. A class may appear in multiple entries. Detection-property names should be unique within each class.

```json
{
  "classPrompts": [
    {
      "classes": ["DOG", "CAT", "HORSE"],
      "prompts": [
        {
          "detectionProperty": "DESCRIPTION",
          "prompt": "Describe the animal color and appearance."
        }
      ]
    }
  ],
  "framePrompts": [
    {
      "detectionProperty": "DESCRIPTION",
      "prompt": "Describe this image."
    }
  ]
}
```

# Outputs

Responses are added to the `detection_properties` of the associated image location. In regular prompt mode, the property name comes from `detectionProperty`. In JSON prompt mode, nested response fields are flattened into Gemini-prefixed detection properties.

# Local vLLM Servers

Two GPU server Dockerfiles provide OpenAI-compatible endpoints for image requests. They are independent of the component image built from `Dockerfile`.

`Dockerfile.vllm` serves `google/gemma-4-12B-it` on port 8000 and supports optional MTP speculative decoding. Set `VLLM_ENABLE_MTP=1` to enable MTP. The default draft model appends `-assistant` to `VLLM_MODEL`; use `VLLM_DRAFT_MODEL` to override it. Mount a populated Hugging Face cache or set `HF_HUB_OFFLINE=0` when model downloads are allowed.

Build it with `docker build -f Dockerfile.vllm -t gemini-image-captioning-vllm .` and publish port 8000 when running it.

`Dockerfile.vllm-diffusiongemma` serves `google/diffusiongemma-26B-A4B-it` on port 8001 using the DiffusionGemma vLLM compatibility shim. Its multimodal request limit allows one image and no video or audio inputs.

Build it with `docker build -f Dockerfile.vllm-diffusiongemma -t gemini-image-captioning-diffusiongemma .` and publish port 8001 when running it.

Configure component jobs with `API=OpenAI`, a matching `MODEL_NAME`, and `OPENAI_BASE_URL=http://server-host:8000/v1` or `http://server-host:8001/v1`.
