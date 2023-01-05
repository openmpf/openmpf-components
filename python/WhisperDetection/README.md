# Overview

This repository contains source code and model data for the OpenMPF Whisper Detection component.
This component uses the OpenAI Whisper model.

# Introduction

This component identifies the language spoken in audio and video clips.

# Input Properties
- `WHISPER_MODEL_SIZE`: Size of the Whisper model. Whisper has tiny, base, small, medium, and large models available for multilingual models. English-only models are available in tiny, base, small, and medium. 
- `WHISPER_MODE`: Determines whether Whisper will perform language detection, speech-to-text transcription, or speech translation. English-only models can only transcribe English audio. 
- `AUDIO_LANGUAGE`: Optional property that indicates the language to use for audio translation or transcription. If left as an empty string, Whisper will automatically detect a single language from the first 30 seconds of audio.

# Output Properties
- `DETECTED_LANGUAGE`: Language with the highest confidence value.
- `DETECTED_LANGUAGE_CONFIDENCE`: The confidence value of the language with the highest confidence.

# Behavior
Some quirks in Whisper's behavior when transcribing or translating audio with multiple languages has been observed. See [whisper_behavior_notes.md](whisper_behavior_notes.md) for more details.