# Overview

This folder sitory contains source code for the OpenMPF Qwen speech summarization component.

This component requires a base image python3.10+ and an mpf_component_api that supports mpf.AllVideoTracksJob.

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

# Outputs

A list of mpf.VideoTracks or mpf.AudioTracks (once supported).

Output[0] will always contain the overall summary of the input, including primary/other topics and entities.
Output[1-n] will be the confidences, reasoning, and name for each of the union of enabled classifiers AND classifiers defined in classifiers.json.

TODO: examples