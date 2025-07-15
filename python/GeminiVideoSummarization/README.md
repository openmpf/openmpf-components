# Overview

This repository contains source code for the OpenMPF Gemini Video Summarization Component.

This component utilizes a text file with a prompt to use on a provided video for analysis. By default, this component does video summarization by utilizing Vertex AI on the Google Cloud Platform(GCP) and their provided models.

# Job Properties

The following are the properties that can be specified for the component. The first two properties have default values, but each one after has to be specified for this component.

- `GENERATION_PROMPT_PATH`: Path to txt file which contains prompt.
- `MODEL_NAME`: The model name for which Gemini model to use.
- `GOOGLE_APPLICATION_CREDENTIALS`: Your gcloud CLI credentials in a json file.
- `PROJECT_ID`: The project ID for your GCP project.
- `BUCKET_NAME`: The GCP bucket that holds the data for processing.
- `LABEL_PREFIX`: The prefix of your labels. ie. Your project name
- `LABEL_USER`: The user using the GCP resources.
- `LABEL_PURPOSE`: The reason for using the GCP resources.

# Custom Prompts

For the default prompt refer to gemini_video_summarization_component/default_prompt.txt.

Set GENERATION_PROMPT_PATH to specify the path to a txt file containing a generation prompt to provide the model.

# Command Line Interface

docker run --rm -v $(pwd):/workspace -e LOG_LEVEL=DEBUG -i openmpf_gemini_video_summarization:latest -M MIME_TYPE=video/mp4 -P GOOGLE_APPLICATION_CREDENTIALS=(INSERT HERE) -P MODEL_NAME=gemini-2.5-pro -P PROJECT_ID=(INSERT HERE) -P LABEL_PREFIX=(INSERT HERE) -P BUCKET_NAME=(INSERT HERE) -P LABEL_USER=(INSERT HERE) -P LABEL_PURPOSE=(INSERT HERE) --pretty test_video.mp4
