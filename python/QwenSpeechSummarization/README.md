# Overview

This folder sitory contains source code for the OpenMPF Qwen speech summarization component.

This component requires a base image python3.10+ and an mpf_component_api that supports mpf.AllVideoTracksJob.

# Inputs

TODO

# Outputs

A list of mpf.VideoTracks or mpf.AudioTracks (once supported).

Output[0] will always contain the overall summary of the input, including primary/other topics and entities.
Output[1-n] will be the confidences, reasoning, and name for each of the union of enabled classifiers AND classifiers defined in classifiers.json.

TODO: examples