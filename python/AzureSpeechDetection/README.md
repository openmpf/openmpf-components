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

- `LANGUAGE`:  The BCP-47 locale to use for transcription. Defaults to `en-US`. A complete list of available locales can be found in Microsoft's [Speech service documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support).

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


| Language                                    | Locale (BCP-47) |
| ------------------------------------------- | --------------- |
| Afrikaans (South Africa)                    | af-ZA           |
| Amharic (Ethiopia)                          | am-ET           |
| Arabic (United Arab Emirates)               | ar-AE           |
| Arabic (Bahrain)                            | ar-BH           |
| Arabic (Algeria)                            | ar-DZ           |
| Arabic (Egypt)                              | ar-EG           |
| Arabic (Israel)                             | ar-IL           |
| Arabic (Iraq)                               | ar-IQ           |
| Arabic (Jordan)                             | ar-JO           |
| Arabic (Kuwait)                             | ar-KW           |
| Arabic (Lebanon)                            | ar-LB           |
| Arabic (Libya)                              | ar-LY           |
| Arabic (Morocco)                            | ar-MA           |
| Arabic (Oman)                               | ar-OM           |
| Arabic (Palestinian Authority)              | ar-PS           |
| Arabic (Qatar)                              | ar-QA           |
| Arabic (Saudi Arabia)                       | ar-SA           |
| Arabic (Syria)                              | ar-SY           |
| Arabic (Tunisia)                            | ar-TN           |
| Arabic (Yemen)                              | ar-YE           |
| Azerbaijani (Latin, Azerbaijan)             | az-AZ           |
| Bulgarian (Bulgaria)                        | bg-BG           |
| Bengali (India)                             | bn-IN           |
| Bosnian (Bosnia and Herzegovina)            | bs-BA           |
| Catalan                                     | ca-ES           |
| Czech (Czechia)                             | cs-CZ           |
| Welsh (United Kingdom)                      | cy-GB           |
| Danish (Denmark)                            | da-DK           |
| German (Austria)                            | de-AT           |
| German (Switzerland)                        | de-CH           |
| German (Germany)                            | de-DE           |
| Greek (Greece)                              | el-GR           |
| English (Australia)                         | en-AU           |
| English (Canada)                            | en-CA           |
| English (United Kingdom)                    | en-GB           |
| English (Ghana)                             | en-GH           |
| English (Hong Kong SAR)                     | en-HK           |
| English (Ireland)                           | en-IE           |
| English (India)                             | en-IN           |
| English (Kenya)                             | en-KE           |
| English (Nigeria)                           | en-NG           |
| English (New Zealand)                       | en-NZ           |
| English (Philippines)                       | en-PH           |
| English (Singapore)                         | en-SG           |
| English (Tanzania)                          | en-TZ           |
| English (United States)                     | en-US           |
| English (South Africa)                      | en-ZA           |
| Spanish (Argentina)                         | es-AR           |
| Spanish (Bolivia)                           | es-BO           |
| Spanish (Chile)                             | es-CL           |
| Spanish (Colombia)                          | es-CO           |
| Spanish (Costa Rica)                        | es-CR           |
| Spanish (Cuba)                              | es-CU           |
| Spanish (Dominican Republic)                | es-DO           |
| Spanish (Ecuador)                           | es-EC           |
| Spanish (Spain)                             | es-ES           |
| Spanish (Equatorial Guinea)                 | es-GQ           |
| Spanish (Guatemala)                         | es-GT           |
| Spanish (Honduras)                          | es-HN           |
| Spanish (Mexico)                            | es-MX           |
| Spanish (Nicaragua)                         | es-NI           |
| Spanish (Panama)                            | es-PA           |
| Spanish (Peru)                              | es-PE           |
| Spanish (Puerto Rico)                       | es-PR           |
| Spanish (Paraguay)                          | es-PY           |
| Spanish (El Salvador)                       | es-SV           |
| Spanish (United States)<sup>1</sup>         | es-US           |
| Spanish (Uruguay)                           | es-UY           |
| Spanish (Venezuela)                         | es-VE           |
| Estonian (Estonia)                          | et-EE           |
| Basque                                      | eu-ES           |
| Persian (Iran)                              | fa-IR           |
| Finnish (Finland)                           | fi-FI           |
| Filipino (Philippines)                      | fil-PH          |
| French (Belgium)                            | fr-BE           |
| French (Canada)<sup>1</sup>                 | fr-CA           |
| French (Switzerland)                        | fr-CH           |
| French (France)                             | fr-FR           |
| Irish (Ireland)                             | ga-IE           |
| Galician                                    | gl-ES           |
| Gujarati (India)                            | gu-IN           |
| Hebrew (Israel)                             | he-IL           |
| Hindi (India)                               | hi-IN           |
| Croatian (Croatia)                          | hr-HR           |
| Hungarian (Hungary)                         | hu-HU           |
| Armenian (Armenia)                          | hy-AM           |
| Indonesian (Indonesia)                      | id-ID           |
| Icelandic (Iceland)                         | is-IS           |
| Italian (Switzerland)                       | it-CH           |
| Italian (Italy)                             | it-IT           |
| Japanese (Japan)                            | ja-JP           |
| Javanese (Latin, Indonesia)                 | jv-ID           |
| Georgian (Georgia)                          | ka-GE           |
| Kazakh (Kazakhstan)                         | kk-KZ           |
| Khmer (Cambodia)                            | km-KH           |
| Kannada (India)                             | kn-IN           |
| Korean (Korea)                              | ko-KR           |
| Lao (Laos)                                  | lo-LA           |
| Lithuanian (Lithuania)                      | lt-LT           |
| Latvian (Latvia)                            | lv-LV           |
| Macedonian (North Macedonia)                | mk-MK           |
| Malayalam (India)                           | ml-IN           |
| Mongolian (Mongolia)                        | mn-MN           |
| Marathi (India)                             | mr-IN           |
| Malay (Malaysia)                            | ms-MY           |
| Maltese (Malta)                             | mt-MT           |
| Burmese (Myanmar)                           | my-MM           |
| Norwegian Bokmål (Norway)                   | nb-NO           |
| Nepali (Nepal)                              | ne-NP           |
| Dutch (Belgium)                             | nl-BE           |
| Dutch (Netherlands)                         | nl-NL           |
| Punjabi (India)                             | pa-IN           |
| Polish (Poland)                             | pl-PL           |
| Pashto (Afghanistan)                        | ps-AF           |
| Portuguese (Brazil)                         | pt-BR           |
| Portuguese (Portugal)                       | pt-PT           |
| Romanian (Romania)                          | ro-RO           |
| Russian (Russia)                            | ru-RU           |
| Sinhala (Sri Lanka)                         | si-LK           |
| Slovak (Slovakia)                           | sk-SK           |
| Slovenian (Slovenia)                        | sl-SI           |
| Somali (Somalia)                            | so-SO           |
| Albanian (Albania)                          | sq-AL           |
| Serbian (Cyrillic, Serbia)                  | sr-RS           |
| Swedish (Sweden)                            | sv-SE           |
| Swahili (Kenya)                             | sw-KE           |
| Swahili (Tanzania)                          | sw-TZ           |
| Tamil (India)                               | ta-IN           |
| Telugu (India)                              | te-IN           |
| Thai (Thailand)                             | th-TH           |
| Turkish (Türkiye)                           | tr-TR           |
| Ukrainian (Ukraine)                         | uk-UA           |
| Urdu (India)                                | ur-IN           |
| Uzbek (Latin, Uzbekistan)                   | uz-UZ           |
| Vietnamese (Vietnam)                        | vi-VN           |
| Chinese (Wu, Simplified)                    | wuu-CN          |
| Chinese (Cantonese, Simplified)             | yue-CN          |
| Chinese (Mandarin, Simplified)              | zh-CN           |
| Chinese (Jilu Mandarin, Simplified)         | zh-CN-shandong  |
| Chinese (Southwestern Mandarin, Simplified) | zh-CN-sichuan   |
| Chinese (Cantonese, Traditional)            | zh-HK           |
| Chinese (Taiwanese Mandarin, Traditional)   | zh-TW           |
| Zulu (South Africa)                         | zu-ZA           |

## Dynamic Speech Selection

The below table describes the component's default behavior when supplied an ISO 639-3 language code by an upstream language identification component in a feed-forward track. For languages with multiple dialects (indicated by an asterisk), a BCP-47 locale was chosen according to internal data, which may not be desirable in all cases. This selection can be altered by editing `acs_speech_component/azure_utils.py`.

If the language code supplied by a feed-forward track is not handled in `acs_speech_component/azure_utils.py`, the component will raise an `INVALID_PROPERTY` exception.

| ISO 639--3 | Language                     | BCP-47 |
| ---------- | ---------------------------- | ------ |
| afr        | Afrikaans                    | af-ZA  |
| amh        | Amharic                      | am-ET  |
| ara        | Arabic                       | ar-EG  |
| aze        | Azerbaijani                  | az-AZ  |
| azj        | North Azerbaijani            | az-AZ  |
| azb        | South Azerbaijani            | az-AZ  |
| ben        | Bengali                      | bn-IN  |
| bul        | Bulgarian                    | bg-BG  |
| bos        | Bosnian                      | bs-BA  |
| cat        | Catalan                      | ca-ES  |
| ces        | Czech                        | cs-CZ  |
| cze        | Czech                        | cs-CZ  |
| cym        | Welsh                        | cy-GB  |
| wel        | Welsh                        | cy-GB  |
| dan        | Danish                       | da-DK  |
| jut        | Jutish                       | da-DK  |
| deu        | German                       | de-DE  |
| gsw        | Swiss German                 | de-CH  |
| bar        | Bavarian                     | de-AT  |
| ell        | Modern Greek (1453-)         | el-GR  |
| eng        | English                      | en-US  |
| est        | Estonian                     | et-EE  |
| ekk        | Standard Estonian            | et-EE  |
| eus        | Basque                       | eu-ES  |
| fas        | Persian                      | fa-IR  |
| fin        | Finnish                      | fi-FI  |
| fil        | Filipino                     | fil-PH |
| fra        | French                       | fr-FR  |
| gle        | Irish                        | ga-IE  |
| glg        | Galician                     | gl-ES  |
| guj        | Gujarati                     | gu-IN  |
| heb        | Hebrew                       | he-IL  |
| hin        | Hindi                        | hi-IN  |
| hrv        | Croatian                     | hr-HR  |
| hun        | Hungarian                    | hu-HU  |
| ita        | Italian                      | it-IT  |
| ind        | Indonesian                   | id-ID  |
| ice        | Icelandic                    | is-IS  |
| isl        | Icelandic                    | is-IS  |
| jav        | Javanese                     | jv-ID  |
| jpn        | Japanese                     | ja-JP  |
| kat        | Georgian                     | ka-GE  |
| kaz        | Kazakh                       | kk-KZ  |
| khm        | Khmer                        | km-KH  |
| kxm        | Northern Khmer               | km-KH  |
| kan        | Kannada                      | kn-IN  |
| kor        | Korean                       | ko-KR  |
| lao        | Lao                          | lo-LA  |
| lit        | Lithuanian                   | lt-LT  |
| lav        | Latvian                      | lv-LV  |
| lvs        | Standard Latvian             | lv-LV  |
| mkd        | Macedonian                   | mk-MK  |
| mya        | Burmese                      | my-MM  |
| mal        | Malayalam                    | ml-IN  |
| mon        | Mongolian                    | mn-MN  |
| khk        | Halh Mongolian               | mn-MN  |
| mvf        | Peripheral Mongolian         | mn-MN  |
| mar        | Marathi                      | mr-IN  |
| zsm        | Standard Malay               | ms-MY  |
| mlt        | Maltese                      | mt-MT  |
| nob        | Norwegian Bokmål             | nb-NO  |
| nep        | Nepali (macrolanguage)       | ne-NP  |
| npi        | Nepali (individual language) | ne-NP  |
| nld        | Dutch                        | nl-NL  |
| pan        | Panjabi                      | pa-IN  |
| pes        | Iranian Persian              | fa-IR  |
| pol        | Polish                       | pl-PL  |
| por        | Portuguese                   | pt-BR  |
| pus        | Pushto                       | ps-AF  |
| pbu        | Northern Pashto              | ps-AF  |
| pst        | Central Pashto               | ps-AF  |
| pbt        | Southern Pashto              | ps-AF  |
| sin        | Sinhala                      | si-LK  |
| ron        | Romanian                     | ro-RO  |
| rus        | Russian                      | ru-RU  |
| slk        | Slovak                       | sk-SK  |
| slv        | Slovenian                    | sl-SI  |
| som        | Somali                       | so-SO  |
| spa        | Spanish                      | es-MX  |
| sqi        | Albanian                     | sq-AL  |
| swa        | Swahili (macrolanguage)      | sw-KE  |
| swe        | Swedish                      | sv-SE  |
| srp        | Serbian                      | sr-RS  |
| tam        | Tamil                        | ta-IN  |
| tel        | Telugu                       | te-IN  |
| tgl        | Tagalog                      | fil-PH |
| tha        | Thai                         | th-TH  |
| tur        | Turkish                      | tr-TR  |
| ukr        | Ukrainian                    | uk-UA  |
| urd        | Urdu                         | ur-IN  |
| uzb        | Uzbek                        | uz-UZ  |
| vie        | Vietnamese                   | vi-VN  |
| cmn        | Mandarin Chinese             | zh-CN  |
| zho        | Chinese                      | zh-CN  |
| yue        | Yue Chinese                  | yue-CN |
| wuu        | Wu Chinese                   | wuu-CN |
| nan        | Min Nan Chinese              | zh-TW  |
| zul        | Zulu                         | zu-ZA  |