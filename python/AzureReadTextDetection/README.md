# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Read Text Detection Component, which utilizes the [Azure Cognitive Services Read Detection REST
endpoint](https://westus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/5d986960601faab4bf452005)
to extract formatted text from documents (PDFs), images, and videos.

**Moving forward, future Read OCR enhancements will fall under the following two services:**
- [Computer Vision Version 4.0 Read OCR Overview](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-ocr) (in preview). [API documentation here](https://centraluseuap.dev.cognitive.microsoft.com/docs/services/unified-vision-apis-public-preview-2023-02-01-preview/operations/61d65934cd35050c20f73ab6).
   - For processing general, non-document images containing text (text in the wild).
- [Form Recognizer Read Model v3.0](https://learn.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/concept-read?view=form-recog-3.0.0) (in preview).
   - For processing scans of documents and clean text.

We are currently reviewing and moving to support Azure's v4.0 Read OCR for processing images containing text.

# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value. Both can also get the value
from environment variables with the same name. If both are provided,
the job property will be used.

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint. Multiple Read API versions exist for text detection
   and can be specified through the URL as follows:

   - `https://{endpoint}/vision/v3.0/read/analyze`            - v3.0 Read OCR analysis.
   - `https://{endpoint}/vision/v3.1/read/analyze`            - v3.1 Read OCR analysis.
   - **`https://{endpoint}/vision/v3.2/read/analyze`          - v3.2 Read OCR analysis. (Final version for v3 Read OCR series)**

   Note that version 3.2 supports a greater number of auto-detected OCR languages, including handwritten text (see [v3.2 supported languages](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/language-support#optical-character-recognition-ocr)).

   Currently listed handwritten language support includes English, Chinese, French, German, Italian, Japanese, Korean, Portuguese, and Spanish.

- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an
  Azure Cognitive Services account.

Optional job properties include:
- `MERGE_LINES`: A boolean toggle to specify whether text detection in a page should appear as separate `LINE` outputs or
   a single `MERGED_LINES` output. The default is to merge lines as an excessive number of detections may be reported
   otherwise.
- `LANGUAGE`:  When provided, disables automatic language detection and processes document based on provided language.
   The language code must be in case-sensitive BCP-47.  Currently, 164 languages are supported, with 9 languages also having handwritten text support: [see language docs here](https://aka.ms/ocr-languages).

# Supported Languages and Associated Codes (Read OCR v3.2)
For tracking purposes, the currently listed 164 supported languages and associated codes for Read OCR v3.2 include:

   | Language                    | Code (optional) | Language                   | Code (optional) |
   | --------------------------- | --------------- | -------------------------- | --------------- |
   | Afrikaans                   | af              | Khasi                      | kha             |
   | Albanian                    | sq              | K'iche'                    | quc             |
   | Angika (Devanagiri)         | anp             | Korean                     | ko              |
   | Arabic                      | ar              | Korku                      | kfq             |
   | Asturian                    | ast             | Koryak                     | kpy             |
   | Awadhi-Hindi (Devanagiri)   | awa             | Kosraean                   | kos             |
   | Azerbaijani (Latin)         | az              | Kumyk (Cyrillic)           | kum             |
   | Bagheli                     | bfy             | Kurdish (Arabic)           | ku-arab         |
   | Basque                      | eu              | Kurdish (Latin)            | ku-latn         |
   | Belarusian (Cyrillic)       | be, be-cyrl     | Kurukh (Devanagiri)        | kru             |
   | Belarusian (Latin)          | be, be-latn     | Kyrgyz (Cyrillic)          | ky              |
   | Bhojpuri-Hindi (Devanagiri) | bho             | Lakota                     | lkt             |
   | Bislama                     | bi              | Latin                      | la              |
   | Bodo (Devanagiri)           | brx             | Lithuanian                 | lt              |
   | Bosnian (Latin)             | bs              | Lower Sorbian              | dsb             |
   | Brajbha                     | bra             | Lule Sami                  | smj             |
   | Breton                      | br              | Luxembourgish              | lb              |
   | Bulgarian                   | bg              | Mahasu Pahari (Devanagiri) | bfz             |
   | Bundeli                     | bns             | Malay (Latin)              | ms              |
   | Buryat (Cyrillic)           | bua             | Maltese                    | mt              |
   | Catalan                     | ca              | Malto (Devanagiri)         | kmj             |
   | Cebuano                     | ceb             | Manx                       | gv              |
   | Chamling                    | rab             | Maori                      | mi              |
   | Chamorro                    | ch              | Marathi                    | mr              |
   | Chhattisgarhi (Devanagiri)  | hne             | Mongolian (Cyrillic)       | mn              |
   | Chinese Simplified          | zh-Hans         | Montenegrin (Cyrillic)     | cnr-cyrl        |
   | Chinese Traditional         | zh-Hant         | Montenegrin (Latin)        | cnr-latn        |
   | Cornish                     | kw              | Neapolitan                 | nap             |
   | Corsican                    | co              | Nepali                     | ne              |
   | Crimean Tatar (Latin)       | crh             | Niuean                     | niu             |
   | Croatian                    | hr              | Nogay                      | nog             |
   | Czech                       | cs              | Northern Sami (Latin)      | sme             |
   | Danish                      | da              | Norwegian                  | no              |
   | Dari                        | prs             | Occitan                    | oc              |
   | Dhimal (Devanagiri)         | dhi             | Ossetic                    | os              |
   | Dogri (Devanagiri)          | doi             | Pashto                     | ps              |
   | Dutch                       | nl              | Persian                    | fa              |
   | English                     | en              | Polish                     | pl              |
   | Erzya (Cyrillic)            | myv             | Portuguese                 | pt              |
   | Estonian                    | et              | Punjabi (Arabic)           | pa              |
   | Faroese                     | fo              | Ripuarian                  | ksh             |
   | Fijian                      | fj              | Romanian                   | ro              |
   | Filipino                    | fil             | Romansh                    | rm              |
   | Finnish                     | fi              | Russian                    | ru              |
   | French                      | fr              | Sadri (Devanagiri)         | sck             |
   | Friulian                    | fur             | Samoan (Latin)             | sm              |
   | Gagauz (Latin)              | gag             | Sanskrit (Devanagari)      | sa              |
   | Galician                    | gl              | Santali(Devanagiri)        | sat             |
   | German                      | de              | Scots                      | sco             |
   | Gilbertese                  | gil             | Scottish Gaelic            | gd              |
   | Gondi (Devanagiri)          | gon             | Serbian (Latin)            | sr, sr-latn     |
   | Greenlandic                 | kl              | Sherpa (Devanagiri)        | xsr             |
   | Gurung (Devanagiri)         | gvr             | Sirmauri (Devanagiri)      | srx             |
   | Haitian Creole              | ht              | Skolt Sami                 | sms             |
   | Halbi (Devanagiri)          | hlb             | Slovak                     | sk              |
   | Hani                        | hni             | Slovenian                  | sl              |
   | Haryanvi                    | bgc             | Somali (Arabic)            | so              |
   | Hawaiian                    | haw             | Southern Sami              | sma             |
   | Hindi                       | hi              | Spanish                    | es              |
   | Hmong Daw (Latin)           | mww             | Swahili (Latin)            | sw              |
   | Ho(Devanagiri)              | hoc             | Swedish                    | sv              |
   | Hungarian                   | hu              | Tajik (Cyrillic)           | tg              |
   | Icelandic                   | is              | Tatar (Latin)              | tt              |
   | Inari Sami                  | smn             | Tetum                      | tet             |
   | Indonesian                  | id              | Thangmi                    | thf             |
   | Interlingua                 | ia              | Tongan                     | to              |
   | Inuktitut (Latin)           | iu              | Turkish                    | tr              |
   | Irish                       | ga              | Turkmen (Latin)            | tk              |
   | Italian                     | it              | Tuvan                      | tyv             |
   | Japanese                    | ja              | Upper Sorbian              | hsb             |
   | Jaunsari (Devanagiri)       | Jns             | Urdu                       | ur              |
   | Javanese                    | jv              | Uyghur (Arabic)            | ug              |
   | Kabuverdianu                | kea             | Uzbek (Arabic)             | uz-arab         |
   | Kachin (Latin)              | kac             | Uzbek (Cyrillic)           | uz-cyrl         |
   | Kangri (Devanagiri)         | xnr             | Uzbek (Latin)              | uz              |
   | Karachay-Balkar             | krc             | Volapük                    | vo              |
   | Kara-Kalpak (Cyrillic)      | kaa-cyrl        | Walser                     | wae             |
   | Kara-Kalpak (Latin)         | kaa             | Welsh                      | cy              |
   | Kashubian                   | csb             | Western Frisian            | fy              |
   | Kazakh (Cyrillic)           | kk-cyrl         | Yucatec Maya               | yua             |
   | Kazakh (Latin)              | kk-latn         | Zhuang                     | za              |
   | Khaling                     | klr             | Zulu                       | zu              |

   ## Handwritten Languages Supported:

   | Language           | Language code (optional) |
   | ------------------ | ------------------------ |
   | English            | en                       |
   | Chinese Simplified | zh-Hans                  |
   | French             | fr                       |
   | German             | de                       |
   | Italian            | it                       |
   | Japanese           | ja                       |
   | Korean             | ko                       |
   | Portuguese         | pt                       |
   | Spanish            | es                       |

# Job Outputs
Currently the component provides text outputs in separate detection tracks.

Detection tracks with `OUTPUT_TYPE` set to `LINE` or `MERGED_LINES` will include `TEXT` results. `MERGED_LINES`
contains `TEXT` output where each line is separated by a newline character.
Confidence scores are calculated as the average confidence of individual words detected in a single `LINE`
or across all `MERGED_LINES`.

Results are organized by `PAGE_NUM` and output indexes (`LINE_NUM`) with 1 as the starting index.
For images, the bounding box information for lines or merged lines is also provided in the detection track.

# Sample Program
`sample_acs_read_detector.py` can be used to quickly test with the Azure
endpoint. It supports images, video (`*.avi`), and PDF inputs only. PDF inputs must contain the `.pdf`
extension to be properly recognized as generic job inputs. Similarly video inputs must contain the ".avi" extension.
Otherwise, the sample code will default to running an image job on the provided input.
