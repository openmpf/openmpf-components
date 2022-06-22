# Overview
This repository contains source code for the OpenMPF Azure Cognitive Services Speech-to-Text Component. This component utilizes the [Azure Cognitive Services Batch Transcription REST endpoint](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/batch-transcription) to transcribe speech from audio and video files.


# Required Job Properties
In order for the component to process any jobs, the job properties listed below must be provided. These properties have no default value, but can be set through environment variables of the same name. If both environment variable and job property are provided, the job property will be used.

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint.
 e.g. `https://virginia.cris.azure.us/api/speechtotext/v2.0/transcriptions`.
 The component has only been tested against v2.0 of the API.

 - `ACS_SUBSCRIPTION_KEY`: A string containing your subscription key for the speech service.

- `ACS_BLOB_CONTAINER_URL`: URL for an Azure Storage Blob container in which to store files during processing.
 e.g. `https://myaccount.blob.core.windows.net/mycontainer`.
 See Microsoft's [documentation on Azure storage](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blob-container-create) for details.

- `ACS_BLOB_SERVICE_KEY`: A string containing your Azure Cognitive Services storage access key.


# Optional Job Properties
The below properties can be optionally provided to alter the behavior of the component.

- `LANGUAGE`:  The locale to use for transcription. Defaults to `en-US`. A complete list of available locales can be found in Microsoft's [Speech service documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support).

- `DIARIZE`: Whether to assign utterances to different speakers. Currently, this component supports only two-speaker diarization. Diarization is enabled by default.

- `CLEANUP`: Whether to delete files from Azure Blob storage container when processing is complete. It is recommended to always keep this enabled, unless it is expected that the same piece of media will be processed multiple times.

- `BLOB_ACCESS_TIME`: The amount of time in minutes for which the Azure Speech service will have access to the file in blob storage.



## Detection Properties
Returned `AudioTrack` objects have the following members in their `detection_properties`:

| Property Key | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SPEAKER_ID` | An integer speaker identifier, indexed from 1. When a job has been segmented by the Workflow Manager, the ID for all utterances will be overwritten by zero, to avoid confusion (as speaker IDs are not consistent between subjobs).                                                                                                                                                                                                                                                                                                                                                                |
| `LONG_SPEAKER_ID` | A unique speaker identifier, of the form "`<start_offset>-<stop_offset>-<#>`, where `<start_offset>` and `<stop_offset>` are integers indicating the segment range (in frame counts for video jobs, milliseconds for audio jobs) for sub-jobs when a job has been segmented by the Workflow Manager. The final `#` portion of the ID is a 1-indexed counter for speaker identity within the indicated segment range. When jobs are not segmented, or not submitted through the Workflow Manager at all, `stop_offset` may instead be `EOF`, indicating that the job extends to the end of the file. |
| `GENDER` | Only present if supplied by an upstream component. The gender of the speaker.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `GENDER_CONFIDENCE` | Only present if supplied by an upstream component. The confidence of the gender classification.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `TRANSCRIPT` | The text of the utterance transcript. Words are space-separated. Olive does not support punctuation beyond that contained in tokens, e.g. acronyms and contractions. If decoding fails, or if the track confidence does not meet the `SPEECH_ACTIVITY_THRESHOLD`, then this field will contain the empty string.                                                                                                                                                                                                                                                                                    |
| `WORD_CONFIDENCES` | The confidence of each word, separated by commas. Each confidence is a real number between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `WORD_SEGMENTS` | The timespan of each word. This is of the format "start_1-stop_1, start_2-stop_2, ..., start_n-stop_n", where each start_x and stop_x is in milliseconds.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `DECODED_LANGUAGE` | The language used to decode this utterance, in BCP-47 format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `SPEAKER_LANGUAGES` | Only present if supplied by an upstream component. All the languages detected by the upstream component's language detection, possibly used to determine the above language used to decode the audio. These languages will be in ISO 639-3 format (see table below for corresponding BCP-47 codes).                                                                                                                                                                                                                                                                                                 |
| `SPEAKER_LANGUAGE_CONFIDENCES` | Only present if supplied by an upstream component. The confidences of each of the above languages, as detected by LID, separated by commas. These values will be either -1, or reals between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                               |


AudioTracks also have the `start_time` and `stop_time` of their associated utterance's voiced segment, and the utterance `confidence`, as returned by Azure.


# Sample Program
`sample_acs_speech_detector.py` can be used to quickly test with the Azure endpoint. Run with the `-h` flag to see accepted command-line arguments.


# Language Identifiers
The following are the BCP-47 codes and their corresponding languages which Azure Speech-to-Text supports. The third column indicates whether the language is also supported by a proprietary upstream language identification component, and the corresponding ISO 639-3 language code that it will pass forward in a feed-forward track. For languages with multiple dialects, a BCP-47 locale was chosen according to internal data, which may not be desirable in all cases. This selection can be altered by editing `acs_speech_component/azure_utils.py`.


| Language                       | BCP-47 | ISO 639-3 |
|--------------------------------|--------|:---------:|
| Afrikaans                      | `af-ZA` |           |
| Amharic                        | `am-ET` |   `AMH`   |
| Arabic (Algeria)               | `ar-DZ` |           |
| Arabic (Bahrain)               | `ar-BH` |           |
| Arabic (Egypt)                 | `ar-EG` |  `ARA`*   |
| Arabic (Iraq)                  | `ar-IQ` |           |
| Arabic (Israel)                | `ar-IL` |           |
| Arabic (Jordan)                | `ar-JO` |           |
| Arabic (Kuwait)                | `ar-KW` |           |
| Arabic (Lebanon)               | `ar-LB` |           |
| Arabic (Libya)                 | `ar-LY` |           |
| Arabic (Morocco)               | `ar-MA` |           |
| Arabic (Oman)                  | `ar-OM` |           |
| Arabic (Palestinian Authority) | `ar-PS` |           |
| Arabic (Qatar)                 | `ar-QA` |           |
| Arabic (Saudi Arabia)          | `ar-SA` |           |
| Arabic (Syria)                 | `ar-SY` |           |
| Arabic (Tunisia)               | `ar-TN` |           |
| Arabic (United Arab Emirates)  | `ar-AE` |           |
| Arabic (Yemen)                 | `ar-YE` |           |
| Bulgarian                      | `bg-BG` |   `BUL`   |
| Burmese                        | `my-MM` |   `MYA`   |
| Catalan                        | `ca-ES` |           |
| Chinese (Cantonese)            | `zh-HK` |  `YUE`*   |
| Chinese (Mandarin)             | `zh-CN` |  `CMN`*   |
| Chinese (Taiwan)               | `zh-TW` |  `NAN`*   |
| Croatian                       | `hr-HR` |           |
| Czech                          | `cs-CZ` |   `CES`   |
| Danish                         | `da-DK` |           |
| Dutch (Belgium)                | `nl-BE` |           |
| Dutch (Netherlands)            | `nl-NL` |           |
| English (Australia)            | `en-AU` |           |
| English (Canada)               | `en-CA` |           |
| English (Ghana)                | `en-GH` |           |
| English (Hong Kong)            | `en-HK` |           |
| English (India)                | `en-IN` |           |
| English (Ireland)              | `en-IE` |           |
| English (Kenya)                | `en-KE` |           |
| English (New Zealand)          | `en-NZ` |           |
| English (Nigeria)              | `en-NG` |           |
| English (Philippines)          | `en-PH` |           |
| English (Singapore)            | `en-SG` |           |
| English (South Africa)         | `en-ZA` |           |
| English (Tanzania)             | `en-TZ` |           |
| English (United Kingdom)       | `en-GB` |           |
| English (United States)        | `en-US` |  `ENG`*   |
| Estonian                       | `et-EE` |           |
| Farsi                          | `fa-IR` |   `PES`   |
| Finnish                        | `fi-FI` |           |
| Filipino                       | `fil-P` |           |
| French (Belgium)               | `fr-BE` |           |
| French (Canada)                | `fr-CA` |           |
| French (France)                | `fr-FR` |  `FRE`*   |
| French (Switzerland)           | `fr-CH` |           |
| German (Austria)               | `de-AT` |           |
| German (Germany)               | `de-DE` |           |
| German (Switzerland)           | `de-CH` |           |
| Greek                          | `el-GR` |   `ELL`   |
| Gujarati                       | `gu-IN` |           |
| Hebrew                         | `he-IL` |           |
| Hindi                          | `hi-IN` |   `HIN`   |
| Hungarian                      | `hu-HU` |           |
| Icelandic                      | `is-IS` |           |
| Indonesian                     | `id-ID` |   `IND`   |
| Irish                          | `ga-IE` |           |
| Italian                        | `it-IT` |           |
| Japanese                       | `ja-JP` |   `JPN`   |
| Javanese                       | `jv-ID` |   `JAV`   |
| Kannada                        | `kn-IN` |           |
| Khmer                          | `km-KH` |           |
| Korean                         | `ko-KR` |   `KOR`   |
| Lao                            | `lo-LA` |   `LAO`   |
| Latvian                        | `lv-LV` |           |
| Lithuanian                     | `lt-LT` |   `LIT`   |
| Macedonian                     | `mk-MK` |   `MKD`   |
| Malay                          | `ms-MY` |           |
| Maltese                        | `mt-MT` |           |
| Marathi                        | `mr-IN` |           |
| Norwegian                      | `nb-NO` |           |
| Polish                         | `pl-PL` |   `POL`   |
| Portuguese (Brazil)            | `pt-BR` |   `POR`   |
| Portuguese (Portugal)          | `pt-PT` |           |
| Romanian                       | `ro-RO` |   `RON`   |
| Russian                        | `ru-RU` |   `RUS`   |
| Serbian                        | `sr-RS` |           |
| Sinhala                        | `si-LK` |           |
| Slovak                         | `sk-SK` |   `SLK`   |
| Slovenian                      | `sl-SI` |           |
| Spanish (Argentina)            | `es-AR` |           |
| Spanish (Bolivia)              | `es-BO` |           |
| Spanish (Chile)                | `es-CL` |           |
| Spanish (Colombia)             | `es-CO` |           |
| Spanish (Costa Rica)           | `es-CR` |           |
| Spanish (Cuba)                 | `es-CU` |           |
| Spanish (Dominican Republic)   | `es-DO` |           |
| Spanish (Ecuador)              | `es-EC` |           |
| Spanish (El Salvador)          | `es-SV` |           |
| Spanish (Equatorial Guinea)    | `es-GQ` |           |
| Spanish (Guatemala)            | `es-GT` |           |
| Spanish (Honduras)             | `es-HN` |           |
| Spanish (Mexico)               | `es-MX` |  `SPA`*   |
| Spanish (Nicaragua)            | `es-NI` |           |
| Spanish (Panama)               | `es-PA` |           |
| Spanish (Paraguay)             | `es-PY` |           |
| Spanish (Peru)                 | `es-PE` |           |
| Spanish (Puerto Rico)          | `es-PR` |           |
| Spanish (Spain)                | `es-ES` |           |
| Spanish (United States)        | `es-US` |           |
| Spanish (Uruguay)              | `es-UY` |           |
| Spanish (Venezuela)            | `es-VE` |           |
| Swahili (Kenya)                | `sw-KE` |  `SWA`*   |
| Swahili (Tanzania)             | `sw-TZ` |           |
| Swedish                        | `sv-SE` |           |
| Tamil                          | `ta-IN` |   `TAM`   |
| Telugu                         | `te-IN` |           |
| Thai                           | `th-TH` |   `THA`   |
| Turkish                        | `tr-TR` |   `TUR`   |
| Ukrainian                      | `uk-UA` |   `UKR`   |
| Uzbek                          | `uz-UZ` |   `UZB`   |
| Vietnamese                     | `vi-VN` |   `VIE`   |
| Zulu                           | `zu-ZA` |   `ZUL`   |