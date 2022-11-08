# Overview

This repository contains source code and model data for the OpenMPF Whisper Detection component.
This component uses the OpenAI Whisper model.

# Introduction

This component identifies the language spoken in audio and video clips.

# Input Properties
- `WHISPER_MODEL_SIZE`: Size of the Whisper model. Whisper has tiny, base, small, medium, and large models available.

# Output Properties
- `DETECTED_LANGUAGE`: Language with the highest confidence value.
- `DETECTED_LANGUAGE_CONFIDENCE`: The confidence value of the language with the highest confidence.