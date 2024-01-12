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

| Property Key                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SPEAKER_ID`              | A unique speaker identifier, of the form "`<start_offset>-<stop_offset>-<#>`, where `<start_offset>` and `<stop_offset>` are integers indicating the segment range (in frame counts for video jobs, milliseconds for audio jobs) for sub-jobs when a job has been segmented by the Workflow Manager. The final `#` portion of the ID is a 1-indexed counter for speaker identity within the indicated segment range. When jobs are not segmented, or not submitted through the Workflow Manager at all, `stop_offset` may instead be `EOF`, indicating that the job extends to the end of the file. |
| `GENDER`                       | Only present if supplied by an upstream component. The gender of the speaker.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `GENDER_CONFIDENCE`            | Only present if supplied by an upstream component. The confidence of the gender classification.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `TRANSCRIPT`                   | The text of the utterance transcript. Words are space-separated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `WORD_CONFIDENCES`             | The confidence of each word, separated by commas. Each confidence is a real number between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `WORD_SEGMENTS`                | The timespan of each word. This is of the format "start_1-stop_1, start_2-stop_2, ..., start_n-stop_n", where each start_x and stop_x is in milliseconds.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ISO_LANGUAGE`                 | The language used to decode this utterance, in ISO 639-3 format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `BCP_LANGUAGE`                 | The language used to decode this utterance, in BCP-47 format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `DECODED_LANGUAGE`             | The language used to decode this utterance, as sent to the Azure endpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `SPEAKER_LANGUAGES`            | Only present if supplied by an upstream component. All the languages detected by the upstream component's language detection, possibly used to determine the above language used to decode the audio. These languages will be in ISO 639-3 format (see table below for corresponding BCP-47 codes).                                                                                                                                                                                                                                                                                                 |
| `SPEAKER_LANGUAGE_CONFIDENCES` | Only present if supplied by an upstream component. The confidences of each of the above languages, as detected by LID, separated by commas. These values will be either -1, or reals between 0 and 1.                                                                                                                                                                                                                                                                                                                                                                                               |
| `MISSING_LANGUAGE_MODELS`      | All languages for which transcription was considered, but which were either invalid ISO 639-3 codes, did not have a corresponding BCP-47 code, or were not supported by the Azure Speech endpoint.                                                                                                                                                                                                                                                                                                                                                                                                  |


AudioTracks also have the `start_time` and `stop_time` of their associated utterance's voiced segment, and the utterance `confidence`, as returned by Azure.


# Sample Program
`sample_acs_speech_detector.py` can be used to quickly test with the Azure endpoint. Run with the `-h` flag to see accepted command-line arguments.


# Language Identifiers
The following are the BCP-47 codes and their corresponding languages which Azure Speech-to-Text supports.


| Language                       | BCP-47 | Language                     | BCP-47  |
|--------------------------------|--------|------------------------------|---------|
| Afrikaans                      | `af-ZA` | Hungarian                    | `hu-HU` |
| Amharic                        | `am-ET` | Icelandic                    | `is-IS` |
| Arabic (Algeria)               | `ar-DZ` | Indonesian                   | `id-ID` |
| Arabic (Bahrain)               | `ar-BH` | Irish                        | `ga-IE` |
| Arabic (Egypt)                 | `ar-EG` | Italian                      | `it-IT` |
| Arabic (Iraq)                  | `ar-IQ` | Japanese                     | `ja-JP` |
| Arabic (Israel)                | `ar-IL` | Javanese                     | `jv-ID` |
| Arabic (Jordan)                | `ar-JO` | Kannada                      | `kn-IN` |
| Arabic (Kuwait)                | `ar-KW` | Khmer                        | `km-KH` |
| Arabic (Lebanon)               | `ar-LB` | Korean                       | `ko-KR` |
| Arabic (Libya)                 | `ar-LY` | Lao                          | `lo-LA` |
| Arabic (Morocco)               | `ar-MA` | Latvian                      | `lv-LV` |
| Arabic (Oman)                  | `ar-OM` | Lithuanian                   | `lt-LT` |
| Arabic (Palestinian Authority) | `ar-PS` | Macedonian                   | `mk-MK` |
| Arabic (Qatar)                 | `ar-QA` | Malay                        | `ms-MY` |
| Arabic (Saudi Arabia)          | `ar-SA` | Maltese                      | `mt-MT` |
| Arabic (Syria)                 | `ar-SY` | Marathi                      | `mr-IN` |
| Arabic (Tunisia)               | `ar-TN` | Norwegian                    | `nb-NO` |
| Arabic (United Arab Emirates)  | `ar-AE` | Polish                       | `pl-PL` |
| Arabic (Yemen)                 | `ar-YE` | Portuguese (Brazil)          | `pt-BR` |
| Bulgarian                      | `bg-BG` | Portuguese (Portugal)        | `pt-PT` |
| Burmese                        | `my-MM` | Romanian                     | `ro-RO` |
| Catalan                        | `ca-ES` | Russian                      | `ru-RU` |
| Chinese (Cantonese)            | `zh-HK` | Serbian                      | `sr-RS` |
| Chinese (Mandarin)             | `zh-CN` | Sinhala                      | `si-LK` |
| Chinese (Taiwan)               | `zh-TW` | Slovak                       | `sk-SK` |
| Croatian                       | `hr-HR` | Slovenian                    | `sl-SI` |
| Czech                          | `cs-CZ` | Spanish (Argentina)          | `es-AR` |
| Danish                         | `da-DK` | Spanish (Bolivia)            | `es-BO` |
| Dutch (Belgium)                | `nl-BE` | Spanish (Chile)              | `es-CL` |
| Dutch (Netherlands)            | `nl-NL` | Spanish (Colombia)           | `es-CO` |
| English (Australia)            | `en-AU` | Spanish (Costa Rica)         | `es-CR` |
| English (Canada)               | `en-CA` | Spanish (Cuba)               | `es-CU` |
| English (Ghana)                | `en-GH` | Spanish (Dominican Republic) | `es-DO` |
| English (Hong Kong)            | `en-HK` | Spanish (Ecuador)            | `es-EC` |
| English (India)                | `en-IN` | Spanish (El Salvador)        | `es-SV` |
| English (Ireland)              | `en-IE` | Spanish (Equatorial Guinea)  | `es-GQ` |
| English (Kenya)                | `en-KE` | Spanish (Guatemala)          | `es-GT` |
| English (New Zealand)          | `en-NZ` | Spanish (Honduras)           | `es-HN` |
| English (Nigeria)              | `en-NG` | Spanish (Mexico)             | `es-MX` |
| English (Philippines)          | `en-PH` | Spanish (Nicaragua)          | `es-NI` |
| English (Singapore)            | `en-SG` | Spanish (Panama)             | `es-PA` |
| English (South Africa)         | `en-ZA` | Spanish (Paraguay)           | `es-PY` |
| English (Tanzania)             | `en-TZ` | Spanish (Peru)               | `es-PE` |
| English (United Kingdom)       | `en-GB` | Spanish (Puerto Rico)        | `es-PR` |
| English (United States)        | `en-US` | Spanish (Spain)              | `es-ES` |
| Estonian                       | `et-EE` | Spanish (United States)      | `es-US` |
| Farsi                          | `fa-IR` | Spanish (Uruguay)            | `es-UY` |
| Finnish                        | `fi-FI` | Spanish (Venezuela)          | `es-VE` |
| Filipino                       | `fil-P` | Swahili (Kenya)              | `sw-KE` |
| French (Belgium)               | `fr-BE` | Swahili (Tanzania)           | `sw-TZ` |
| French (Canada)                | `fr-CA` | Swedish                      | `sv-SE` |
| French (France)                | `fr-FR` | Tamil                        | `ta-IN` |
| French (Switzerland)           | `fr-CH` | Telugu                       | `te-IN` |
| German (Austria)               | `de-AT` | Thai                         | `th-TH` |
| German (Germany)               | `de-DE` | Turkish                      | `tr-TR` |
| German (Switzerland)           | `de-CH` | Ukrainian                    | `uk-UA` |
| Greek                          | `el-GR` | Uzbek                        | `uz-UZ` |
| Gujarati                       | `gu-IN` | Vietnamese                   | `vi-VN` |
| Hebrew                         | `he-IL` | Zulu                         | `zu-ZA` |
 | Hindi                          | `hi-IN` |                              |         |

## Dynamic Speech Selection

The below table describes the component's default behavior when supplied an ISO 639-3 language code by an upstream language identification component in a feed-forward track. For languages with multiple dialects (indicated by an asterisk), a BCP-47 locale was chosen according to internal data, which may not be desirable in all cases. This selection can be altered by editing `acs_speech_component/azure_utils.py`.

If the language code supplied by a feed-forward track is not handled in `acs_speech_component/azure_utils.py`, the component will raise an `INVALID_PROPERTY` exception.

| ISO 639-3 | Language            | BCP-47   |
|:---------:|---------------------|----------|
|   `AMH`   | Amharic             | `am-ET`  |
|   `ARA`   | Arabic              | `ar-EG`* |
|   `BUL`   | Bulgarian           | `bg-BG`  |
|   `CES`   | Czech               | `cs-CZ`  |
|   `CMN`   | Chinese (Mandarin)  | `zh-CN`* |
|   `ELL`   | Greek               | `el-GR`  |
|   `ENG`   | English             | `en-US`* |
|   `FRE`   | French              | `fr-FR`* |
|   `HIN`   | Hindi               | `hi-IN`  |
|   `IND`   | Indonesian          | `id-ID`  |
|   `JAV`   | Javanese            | `jv-ID`  |
|   `JPN`   | Japanese            | `ja-JP`  |
|   `KOR`   | Korean              | `ko-KR`  |
|   `LAO`   | Lao                 | `lo-LA`  |
|   `LIT`   | Lithuanian          | `lt-LT`  |
|   `MKD`   | Macedonian          | `mk-MK`  |
|   `MYA`   | Burmese             | `my-MM`  |
|   `NAN`   | Chinese (Taiwan)    | `zh-TW`* |
|   `PES`   | Farsi               | `fa-IR`  |
|   `POL`   | Polish              | `pl-PL`  |
|   `POR`   | Portuguese          | `pt-BR`  |
|   `RON`   | Romanian            | `ro-RO`  |
|   `RUS`   | Russian             | `ru-RU`  |
|   `SLK`   | Slovak              | `sk-SK`  |
|   `SPA`   | Spanish             | `es-MX`* |
|   `SWA`   | Swahili             | `sw-KE`* |
|   `TAM`   | Tamil               | `ta-IN`  |
|   `THA`   | Thai                | `th-TH`  |
|   `TUR`   | Turkish             | `tr-TR`  |
|   `UKR`   | Ukrainian           | `uk-UA`  |
|   `UZB`   | Uzbek               | `uz-UZ`  |
|   `VIE`   | Vietnamese          | `vi-VN`  |
|   `YUE`   | Chinese (Cantonese) | `zh-HK`* |
|   `ZUL`   | Zulu                | `zu-ZA`  |
