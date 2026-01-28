# Overview

This repository contains source code for the OpenMPF Gemini Video Summarization Component.

This component analyzes a video with a summary and event timeline using Vertex AI on the Google Cloud Platform(GCP) and their provided models. By default, this component does a general video summarization and event timeline.

# Job Properties

The following are the properties that can be specified for the component. The first eight properties have default values, but the other properties have to be specified for this component.

- `MODEL_NAME`: The model name for which Gemini model to use.
- `GENERATION_PROMPT_PATH`: Path to txt file which contains prompt.
- `GENERATION_MAX_ATTEMPTS`: The number of attempts to get a valid JSON response from the model.
- `TIMELINE_CHECK_TARGET_THRESHOLD`: Specifies the number of seconds that video events can occur before or after video bounds. If exceeded, another attempt will be made to generate the output. Set to -1 to disable check.
- `MERGE_TRACKS`: In the context of videos, when set to true, attempt to merge tracks from the entire video.
- `TARGET_SEGMENT_LENGTH`: The length of segments the video is divided into.
- `VFR_TARGET_SEGMENT_LENGTH`: The length of segments the video is divided into.
- `PROCESS_FPS`: The amount of frames to process per second. The FPS processing limit for Gemini is (0.0, 24.0]
- `ENABLE_TIMELINE`: When set to 1, this enables the creation of timelines. If there is no custom prompt file, it will use default_prompt.txt. When set to 0, this disables timelines. If there is no custom prompt file, it will use default_prompt_no_tl.txt.

REQUIRED PROPERTIES:
- `PROJECT_ID`: The project ID for your GCP project.
- `BUCKET_NAME`: The GCP bucket that holds the data for processing.
- `LABEL_PREFIX`: The prefix of your labels. ie. Your project name
- `LABEL_USER`: The user using the GCP resources.
- `LABEL_PURPOSE`: The reason for using the GCP resources.
- `GOOGLE_APPLICATION_CREDENTIALS`: Your gcloud CLI credentials in a json file.

# Custom Prompts

For the default prompt with timelines enabled refer to gemini_video_summarization_component/data/default_prompt.txt. 

For the default prompt WITHOUT timelines enabled refer to gemini_video_summarization_component/data/default_prompt_no_tl.txt.

Set GENERATION_PROMPT_PATH to specify the path to a txt file containing a generation prompt to provide the model. 

When making a custom prompt or altering the default prompt with timelines enabled, it is required that the parts about timestamp formatting, timestamp offsets and the JSON structured output are included in your prompt.

# GCP Certificate

Using this component required access to Vertex AI. Which means adding a GCP certificate required. 
Set GOOGLE_APPLICATION_CREDENTIALS to specify the path to the GCP certificate JSON file.

# Timestamps

When videos exceed 2 minutes in length, timestamps for events become more inaccurate. For accurate timestamps, it is recommended to keep TARGET_SEGMENT_LENGTH and VFR_TARGET_SEGMENT_LENGTH at 120.
If you'd prefer more cohesive summaries over timeline accuracy, you can pass the whole video as one segment by setting TARGET_SEGMENT_LENGTH and VFR_TARGET_SEGMENT_LENGTH to -1. 
Keep in mind the maximum length of a video that can be processed is 45 minutes(2700s). 
This means TARGET_SEGMENT_LENGTH and VFR_TARGET_SEGMENT_LENGTH both have a max of 2700 and are REQUIRED to be set for videos longer than 45 minutes.

To prevent further inaccuracies, Gemini does timestamps best when formatting them in MM:SS format. This means the component does conversions between that format to seconds and back itself. 
So, if altering the prompt, leave in instructions about timestamp formatting.

Another cause of timestamp inaccuracies is the model you are using. Not only does descriptions and summaries lower in quality with Gemini flash models, the timestamps are also become more inaccurate.
For the most accurate timestamps, use segmentation and the latest Gemini pro model.

# Docker Container
  gemini-video-summarization:
    <<: *detection-component-base
    image: <IMAGE>
    volumes:
      - host_directory/cert_name.json:container_directory/cert_name.json:ro # Mount the GCP file from your localhost to the container
      - host_directory/prompt_file.txt:container_directory/prompt_file.txt:ro # OPTIONAL: Mount the custom prompt file from your localhost to the container
      - shared_data:/opt/mpf/share
    environment:
      - MPF_PROP_PROJECT_ID=<PROJECT ID>
      - MPF_PROP_BUCKET_NAME=<BUCKET NAME>
    ... # Add more properties here
      - MPF_PROP_GOOGLE_APPLICATION_CREDENTIALS=container_directory/cert_name.json
      - MPF_PROP_GENERATION_PROMPT_PATH=container_directory/prompt_file.txt # OPTIONAL, but needed if mounted a custom prompt file
