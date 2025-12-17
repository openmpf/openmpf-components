# Overview

This folder contains source code for the OpenMPF Qwen speech summarization component.

This component requires a base image python3.10+ and an mpf_component_api that supports mpf.AllVideoTracksJob.

We have tested Qwen/Qwen3-30B-A3B-Instruct-2507 on an 80GB card and Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 on a 40GB card. Both seem quite viable.

If you are daring, any openai-compatible API could be substituted for VLLM and any model could replace Qwen3-30B BUT these scenarios are untested
and your mileage may vary.

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

- CLASSIFIERS_FILE: when set to an absolute path (with a valid classifiers.json in a volume mounted such that the file is at the specified path), will replace the default classifiers.json
- CLASSIFIERS_LIST: Either "ALL", or a comma-separated list of specific names of the "Classifier" fields of defined classifiers
- PROMPT_TEMPLATE: if set, will replace the packaged `templates/prompt.jinja` with one read from this location. Must include self-recursive summarization instructions and the jinja templates `{{ classifiers }}` and `{{ input }}`.

# Docker build-args

- VLLM_MODEL: if building Dockerfile.vllm for vllm (which downloads the model during docker build), this is the ONLY model that your qwen_speech_summarization_component will be able to use.

NOTE: if you have an internet connection at runtime, you may use the image `vllm/vllm-openai:latest` directly in lieu of building Dockerfile.vllm. We do not support this arrangement BUT it is possible with the right command on the docker service.

# Environment variables

- VLLM_MODEL: must MATCH the model name being served by vllm OR be available at whichver openai-api-compatible API you choose to talk to.
- VLLM_URI: the base_url of the openai-api-compatible API providing access to your model. If your vllm service is named vllm, then this would need to be `http://vllm:11434/v1`.
- MODEL_MAX_LEN should be defined on both the qwen container AND the vllm container. It is the maximum input+output token count you can fit into your VRAM.
- INPUT_TOKEN_CHUNK_SIZE should be about 20%-30% of your MODEL_MAX_LEN, and is the token size that your input will be split into during chunking before making a series of calls to the LLM.
- INPUT_CHUNK_TOKEN_OVERLAP should be small and constant. If it is too small, there will be no overlap between chunks, which could negatively impact performance with huge input tracks.

# Outputs

A list of mpf.VideoTracks or mpf.AudioTracks (once supported).

Output[0] will always contain the overall summary of the input, including primary/other topics and entities.
Output[1-n] will be the confidences, reasoning, and name for each of the union of enabled classifiers AND classifiers defined in classifiers.json.