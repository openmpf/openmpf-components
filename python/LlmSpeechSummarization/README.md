# Overview

The OpenMPF LLM Speech Summarization Component summarizes feed-forward video and audio tracks for speech detections.

# Details

This folder contains source code for the OpenMPF LLM Speech Summarization Component.

This component requires a base image python3.10+ and an mpf_component_api that supports mpf.AllVideoTracksJob and mpf.AllAudioTracksJob.

The component is designed and tested for use with Azure OpenAI or self-hosted Qwen models (via vLLM). It uses the OpenAI-compatible API protocol, so other compatible endpoints may work but are unsupported. The necessary configuration differs by whether you use Azure or a self-hosted LLM, and the differences are described further in [Configuration By Deployment](#configuration-by-deployment).

## vLLM and Qwen usage
We have tested Qwen/Qwen3-30B-A3B-Instruct-2507 on an 80GB card and Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 on a 40GB card. Both seem quite viable.

Any openai-compatible API could be potentially be substituted for VLLM and any model could replace Qwen3-30B BUT these scenarios are untested and your mileage may vary.

In either case, the component assumes anonymous access to the openai-api-compatible endpoint that performs the summarization.

# Inputs

- classifiers.json: contains a definition of subjects of interest to score with a low 0-1 confidence if the input DOES NOT include the defined classifier OR high if it does

```json
[
    {
        "Classifier": "Major League Baseball",
        "Definition": "discussions regarding major league baseball teams, professional baseball players, and baseball stadiums",
        "Items of Interest": "Baseball fields, baseball teams, baseball players, baseballs, baseball bats, baseball hats"
    }
]
```

# Properties

- `CLASSIFIERS_FILE`: when set to an absolute path (with a valid classifiers.json in a volume mounted such that the file is at the specified path), will replace the default classifiers.json
- `CLASSIFIERS_LIST`: Either "ALL", or a comma-separated list of specific names of the "Classifier" fields of defined classifiers
- `PROMPT_TEMPLATE`: if set, will replace the packaged `templates/prompt.jinja` with one read from this location. Must include self-recursive summarization instructions and the jinja templates `{{ classifiers }}` and `{{ input }}`.
- `INPUT_TOKEN_CHUNK_SIZE` should be about 20%-30% of your `MAX_MODEL_LEN`, and is the token size that your input will be split into during chunking before making a series of calls to the LLM.
- `INPUT_CHUNK_TOKEN_OVERLAP` should be small and constant. If it is too small, there will be no overlap between chunks, which could negatively impact performance with huge input tracks.

## Configuration by Deployment

### Azure OpenAI

| Property | Example Value | Notes |
|---|---|---|
| `AZURE_CLIENT` | `true` | Must be set to use the Azure OpenAI client |
| `AZURE_API_VERSION` | `2025-01-01-preview` | Must be set to use the Azure OpenAI client |
| `API_URI` | `https://my-resource.openai.azure.com/openai/deployments/my-gpt-41-mini` | Your Azure OpenAI endpoint URL |
| `API_HEALTH_URI` | *(leave blank)* | Azure is a managed service; health checking is not applicable |
| `API_TOKEN` | `abc123...` | Your Azure API key |
| `LLM_MODEL` | `my-gpt-4.1-mini` | Must match your Azure deployment name |
| `TOKENIZER_MODEL` | `openai/gpt-oss-120b` | Tokenizer used for input chunking; use one compatible with your deployed model |
| `MAX_MODEL_LEN` | `38000` | Set to match the context window of your deployed model |

### Self-Hosted (Qwen via vLLM)

| Property | Example Value | Notes |
|---|---|---|
| `AZURE_CLIENT` | `false` | Omit or set false for non-Azure endpoints |
| `API_URI` | `http://my-vllm-server:11434/v1` | Base URL of your vLLM server |
| `API_HEALTH_URI` | `http://my-vllm-server:11434/health` | Optional; component will wait for this endpoint to return 200 before proceeding |
| `API_TOKEN` | `any-value` | vLLM requires a token but does not validate it by default |
| `LLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Must match the model ID as loaded in vLLM |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Must match the model ID; used for token counting |
| `MAX_MODEL_LEN` | `45000` | Should match the `--max-model-len` value set when launching vLLM. We have tried 45000 for the -FP8 model and 120000 for the nonquantized model on a 40GB and 80GB card, respectively. |

# vLLM Docker build-args

- `VLLM_MODEL`: The model to download and bake into the image. This is the only model the component will be able to use with this image. Defaults to `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`.

Example build command:
```sh
docker build \
  --build-arg VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  -f Dockerfile.vllm \
  -t openmpf_llm_speech_summarization_server:latest \
  .
```

NOTE: if you have an internet connection at runtime, you may use the image `vllm/vllm-openai:latest` directly in lieu of building Dockerfile.vllm. We do not support this arrangement BUT it is possible with the right command on the docker service.

# Outputs

A list of mpf.VideoTracks or mpf.AudioTracks.

Track[0] will always contain the overall summary of the input, including primary/other topics and entities.

Track[1-n] will be the confidences, reasoning, and name for each of the intersection of enabled classifiers AND classifiers defined in classifiers.json.

If you want to control which entities are included in the output, you would need to modify the `EntitiesObject` in [schema.py](./llm_speech_summarization_component/schema.py).
