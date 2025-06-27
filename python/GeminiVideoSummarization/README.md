# Overview

This repository contains source code for the OpenMPF Gemini Video Summarization Component.

This component utilizes a text file with a prompt to use on a provided video for analysis. By default, this component does video summarization by utilizing Google AI Studio and their provided models.

# Job Properties

The following are the properties that can be specified for the component. Each property has a default value and so none of them necessarily need to be specified for processing jobs.

- `PROMPT_CONFIGURATION_PATH`: Path to txt file which contains prompt.
- `GEMINI_API_KEY`: Your API key to send requests to Google Gemini.
- `MODEL_NAME`: The model name for which Gemini model to use.
- `PROCESS_FPS`: The frame rate in which the video will be processed.

# Outputs
...

# TODO

- Implement tracks + timelines
- Add more functionality from the LlamaSummarizationComponent
- Update output to match the LlamaSummarizationComponent