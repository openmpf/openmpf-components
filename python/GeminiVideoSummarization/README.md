# Overview

This repository contains source code for the OpenMPF Gemini Video Summarization Component.

This component analyzes a video with a summary and event timeline using Vertex AI on the Google Cloud Platform(GCP) and their provided models. By default, this component does a general video summarization and event timeline.

# Job Properties

The following are the properties that can be specified for the component. The first six properties have default values, but the other properties have to be specified for this component.

- `MODEL_NAME`: The model name for which Gemini model to use.
- `GENERATION_PROMPT_PATH`: Path to txt file which contains prompt.
- `GOOGLE_APPLICATION_CREDENTIALS`: Your gcloud CLI credentials in a json file.
- `GENERATION_MAX_ATTEMPTS`: The number of attempts to get a valid JSON response from the model.
- `TIMELINE_CHECK_TARGET_THRESHOLD`: Specifies the number of seconds that video events can occur before or after video bounds. If exceeded, another attempt will be made to generate the output. Set to -1 to disable check.
- `MERGE_TRACKS`: In the context of videos, when set to true, attempt to merge tracks from the entire video.
- `PROJECT_ID`: The project ID for your GCP project.
- `BUCKET_NAME`: The GCP bucket that holds the data for processing.
- `LABEL_PREFIX`: The prefix of your labels. ie. Your project name
- `LABEL_USER`: The user using the GCP resources.
- `LABEL_PURPOSE`: The reason for using the GCP resources.

# Custom Prompts

For the default prompt refer to gemini_video_summarization_component/data/default_prompt.txt. 

Set GENERATION_PROMPT_PATH to specify the path to a txt file containing a generation prompt to provide the model. 

When making a custom prompt or altering the default prompt. It is required that the parts about timestamp formatting, timestamp conversion and the JSON structured output are included in your prompt.
(ie. If you are changing the default prompt, change line 1 only)

# GCP Certificate

For the file to enter your GCP credentials refer to gemini_video_summarization_component/data/gcpCert.json. 

Set GOOGLE_APPLICAITION_CREDENTIALS to specify a path to a json file with your credentials if you have it in another location. 

# Command Line Interface

docker run --rm -i openmpf_gemini_video_summarization:latest -t video --end (INSERT HERE) -M MIME_TYPE=video/mp4 -P GOOGLE_APPLICATION_CREDENTIALS=(INSERT HERE) -P PROJECT_ID=(INSERT HERE) -P LABEL_PREFIX=(INSERT HERE) -P BUCKET_NAME=(INSERT HERE) -P LABEL_USER=(INSERT HERE) -P LABEL_PURPOSE=(INSERT HERE) test_video.mp4 > output.json
