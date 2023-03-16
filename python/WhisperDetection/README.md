# Overview

This repository contains source code and model data for the OpenMPF Whisper Detection component.
This component uses the OpenAI Whisper model.

# Introduction

This component identifies the language spoken in audio and video clips.

# Input Properties
- `WHISPER_MODEL_SIZE`: Size of the Whisper model. Whisper has `tiny`, `base`, `small`, `medium`, and `large` models available for multilingual models. English-only models are available in `tiny`, `base`, `small`, and `medium`. 
- `WHISPER_MODEL_LANG`: Whisper has English-only models and multilingual models. Set to `en` for English-only models and `multi` for multilingual models.
- `WHISPER_MODE`: Determines whether Whisper will perform language detection, speech-to-text transcription, or speech translation. English-only models can only transcribe English audio. Set to `0` for spoken language detection, `1` for speech-to-text transcription, and `2` for speech translation.
- `AUDIO_LANGUAGE`: Optional property that indicates the language to use for audio translation or transcription. If left as an empty string, Whisper will automatically detect a single language from the first 30 seconds of audio.

# Output Properties
- `DETECTED_LANGUAGE`: Language with the highest confidence value.
- `DETECTED_LANGUAGE_CONFIDENCE`: The confidence value of the language with the highest confidence.
- `TRANSCRIPT`: Returns transcript of audio for transcription and translation runs.
- `TRANSLATED_AUDIO`: Returns the translated text for translated audio runs.

# Behavior
Some quirks in Whisper's behavior when transcribing or translating audio with multiple languages has been observed. See [whisper_behavior_notes.md](whisper_behavior_notes.md) for more details.

# Language Identifiers
The following are the ISO 639-1 codes, the ISO 639-2 codes, and their corresponding languages which Whisper can translate to English.

All translations are to English.

| ISO-639-1 | ISO-639-2 | Language         |
| --- |---|------------------|
| `af` | `afr` | Afrikaans        |
| `ar` | `ara` | Arabic           |
| `hy` | `hye` | Armenian         |
| `az` | `aze` | Azerbaijani      |
| `be` | `bel` | Belarusian       |
| `bs` | `bos` | Bosnian          |
| `bg` | `bul` | Bulgarian        |
| `ca` | `cat` | Catalan          |
| `zh` | `zho` | Chinese          |
| `hr` | `hrv` | Croatian         |
| `cs` | `ces` | Czech            |
| `da` | `dan` | Danish           |
| `nl` | `nld` | Dutch            |
| `en` | `eng` | English          |
| `fi` | `fin` | Finnish          |
| `fr` | `fra` | French           |
| `gl` | `glg` | Galician         |
| `de` | `deu` | German           |
| `el` | `ell` | Greek            |
| `he` | `heb` | Hebrew           |
| `hi` | `hin` | Hindi            |
| `hu` | `hun` | Hungarian        |
| `is` | `isl` | Icelandic        |
| `id` | `ind` | Indonesian       |
| `it` | `ita` | Italian          |
| `ja` | `jpn` | Japanese         |
| `kn` | `kan` | Kannada          |
| `kk` | `kaz` | Kazakh           |
| `ko` | `kor` | Korean           |
| `lv` | `lav` | Latvian          |
| `lt` | `lit` | Lithuanian       |
| `mk` | `mkd` | Macedonian       |
| `ms` | `msa` | Malay            |
| `mi` | `mri` | Maori            |
| `mr` | `mar` | Marathi          |
| `ne` | `nep` | Nepali           |
| `no` | `nor` | Norwegian        |
| `fa` | `fas` | Persian          |
| `pl` | `pol` | Polish           |
| `pt` | `por` | Portuguese       |
| `ro` | `ron` | Romanian         |
| `ru` | `rus` | Russian          |
| `sr` | `srp` | Serbian          |
| `sk` | `slk` | Slovak           |
| `sl` | `slv` | Slovenian        |
| `es` | `spa` | Spanish          |
| `sw` | `swa` | Swahili          |
| `sv` | `swe` | Swedish          |
| `tl` | `tgl` | Tagalog          |
| `ta` | `tam` | Tamil            |
| `th` | `tha` | Thai             |
| `tr` | `tur` | Turkish          |
| `uk` | `ukr` | Ukrainian        |
| `ur` | `urd` | Urdu             |
| `vi` | `vie` | Vietnamese       |
| `cy` | `cym` | Welsh            |