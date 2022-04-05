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

# Sample Program
`sample_acs_speech_detector.py` can be used to quickly test with the Azure endpoint. Run with the `-h` flag to see accepted command-line arguments.


| BCP | Language | ISO |
| `af-ZA` | Afrikaans |  |
| `am-ET` | Amharic | `AMH` |
| `ar-AE` | Arabic (United Arab Emirates) |  |
| `ar-BH` | Arabic (Bahrain) |  |
| `ar-DZ` | Arabic (Algeria) |  |
| `ar-EG` | Arabic (Egypt) |  |
| `ar-IL` | Arabic (Israel) |  |
| `ar-IQ` | Arabic (Iraq) |  |
| `ar-JO` | Arabic (Jordan) |  |
| `ar-KW` | Arabic (Kuwait) |  |
| `ar-LB` | Arabic (Lebanon) |  |
| `ar-LY` | Arabic (Libya) |  |
| `ar-MA` | Arabic (Morocco) |  |
| `ar-OM` | Arabic (Oman) |  |
| `ar-PS` | Arabic (Palestinian Authority) |  |
| `ar-QA` | Arabic (Qatar) |  |
| `ar-SA` | Arabic (Saudi Arabia) | `ARA` |
| `ar-SY` | Arabic (Syria) |  |
| `ar-TN` | Arabic (Tunisia) |  |
| `ar-YE` | Arabic (Yemen) |  |
| `bg-BG` | Bulgarian | `BUL` |
| `ca-ES` | Catalan |  |
| `cs-CZ` | Czech | `CES` |
| `da-DK` | Danish |  |
| `de-AT` | German (Austria) |  |
| `de-CH` | German (Switzerland) |  |
| `de-DE` | German (Germany) |  |
| `el-GR` | Greek | `ELL` |
| `en-AU` | English (Australia) |  |
| `en-CA` | English (Canada) |  |
| `en-GB` | English (United Kingdom) |  |
| `en-GH` | English (Ghana) |  |
| `en-HK` | English (Hong Kong) |  |
| `en-IE` | English (Ireland) |  |
| `en-IN` | English (India) |  |
| `en-KE` | English (Kenya) |  |
| `en-NG` | English (Nigeria) |  |
| `en-NZ` | English (New Zealand) |  |
| `en-PH` | English (Philippines) |  |
| `en-SG` | English (Singapore) |  |
| `en-TZ` | English (Tanzania) |  |
| `en-US` | English (United States) | `ENG` |
| `en-ZA` | English (South Africa) |  |
| `es-AR` | Spanish (Argentina) |  |
| `es-BO` | Spanish (Bolivia) |  |
| `es-CL` | Spanish (Chile) |  |
| `es-CO` | Spanish (Colombia) |  |
| `es-CR` | Spanish (Costa Rica) |  |
| `es-CU` | Spanish (Cuba) |  |
| `es-DO` | Spanish (Dominican Republic) |  |
| `es-EC` | Spanish (Ecuador) |  |
| `es-ES` | Spanish (Spain) |  |
| `es-GQ` | Spanish (Equatorial Guinea) |  |
| `es-GT` | Spanish (Guatemala) |  |
| `es-HN` | Spanish (Honduras) |  |
| `es-MX` | Spanish (Mexico) | `SPA` |
| `es-NI` | Spanish (Nicaragua) |  |
| `es-PA` | Spanish (Panama) |  |
| `es-PE` | Spanish (Peru) |  |
| `es-PR` | Spanish (Puerto Rico) |  |
| `es-PY` | Spanish (Paraguay) |  |
| `es-SV` | Spanish (El Salvador) |  |
| `es-US` | Spanish (United States) |  |
| `es-UY` | Spanish (Uruguay) |  |
| `es-VE` | Spanish (Venezuela) |  |
| `et-EE` | Estonian |  |
| `fa-IR` | Farsi | `PES` |
| `fi-FI` | Finnish |  |
| `fil-P` | Filipino |  |
| `fr-BE` | French (Belgium) |  |
| `fr-CA` | French (Canada) |  |
| `fr-CH` | French (Switzerland) |  |
| `fr-FR` | French (France) | `FRE` |
| `ga-IE` | Irish |  |
| `gu-IN` | Gujarati |  |
| `he-IL` | Hebrew |  |
| `hi-IN` | Hindi | `HIN` |
| `hr-HR` | Croatian |  |
| `hu-HU` | Hungarian |  |
| `id-ID` | Indonesian | `IND` |
| `is-IS` | Icelandic |  |
| `it-IT` | Italian |  |
| `ja-JP` | Japanese | `JPN` |
| `jv-ID` | Javanese | `JAV` |
| `km-KH` | Khmer |  |
| `kn-IN` | Kannada |  |
| `ko-KR` | Korean | `KOR` |
| `lo-LA` | Lao | `LAO` |
| `lt-LT` | Lithuanian | `LIT` |
| `lv-LV` | Latvian |  |
| `mk-MK` | Macedonian | `MKD` |
| `mr-IN` | Marathi |  |
| `ms-MY` | Malay |  |
| `mt-MT` | Maltese |  |
| `my-MM` | Burmese | `MYA` |
| `nb-NO` | Norwegian |  |
| `nl-BE` | Dutch (Belgium) |  |
| `nl-NL` | Dutch (Netherlands) |  |
| `pl-PL` | Polish | `POL` |
| `pt-BR` | Portuguese (Brazil) | `POR` |
| `pt-PT` | Portuguese (Portugal) |  |
| `ro-RO` | Romanian |`RON`  |
| `ru-RU` | Russian | `RUS` |
| `si-LK` | Sinhala |  |
| `sk-SK` | Slovak | `SLK` |
| `sl-SI` | Slovenian |  |
| `sr-RS` | Serbian |  |
| `sv-SE` | Swedish |  |
| `sw-KE` | Swahili (Kenya) | `SWA` |
| `sw-TZ` | Swahili (Tanzania) |  |
| `ta-IN` | Tamil | `TAM` |
| `te-IN` | Telugu |  |
| `th-TH` | Thai | `THA` |
| `tr-TR` | Turkish | `TUR` |
| `uk-UA` | Ukrainian | `UKR` |
| `uz-UZ` | Uzbek | `UZB` |
| `vi-VN` | Vietnamese | `VIE` |
| `zh-CN` | Chinese (Mandarin) | `CMN`, `NAN` |
| `zh-HK` | Chinese (Cantonese) | `YUE` |
| `zh-TW` | Chinese (Taiwan) |  |
| `zu-ZA` |  | `ZUL` |
