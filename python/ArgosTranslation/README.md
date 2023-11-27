# Overview

This repository contains source code for the OpenMPF Argos Translation Component.

This component translates the input text from a given source language to English. The source language can be provided as a job property, or be indicated in the detection properties from a feed-forward track.


# Job Properties
The below properties can be optionally provided to alter the behavior of the component.

- `FEED_FORWARD_PROP_TO_PROCESS`: Controls which properties of the feed-forward track or detection are considered for translation. This should be a comma-separated list of property names (default: `"TEXT,TRANSCRIPT"`). The first named property in the list that is present is translated. At most, one property will be translated.

- `LANGUAGE_FEED_FORWARD_PROP`: As above, a comma-separated list of property names (default: `"DECODED_LANGUAGE,LANGUAGE"`), which indicate which property of the feed-forward track or detection may be used to determine the source language of the text to be translated. This language is expected to be provided as an ISO 639-1 or and ISO 639-2 language code.

- `DEFAULT_SOURCE_LANGUAGE`: The default source language to use if none of the property names listed in `LANGUAGE_FEED_FORWARD_PROP` are present in a feed-forward track or detection. This language is used when running a generic job with a raw text file (hence no feed-forward tracks).


# Language Identifiers
The following are the ISO 639-1 codes, the ISO 639-2 codes, and their corresponding languages which Argos Translate version 1.7.0 can translate to English.

All translations are either to English or from English. When trying to translate from one non-English language to another, Argos will automatically pivot between languages using the currently installed packages. For example, for Spanish->French Argos would pivot from Spanish->English to English->French. This is associated with a drop in accuracy and increase in runtime.

Language packages are downloaded dynamically as needed. In addition, when building a Docker image the Dockerfile pre-installs German, French, Russian, and Spanish.

Note: Argos has two different models for Chinese Simplified and Traditional. Using the wrong one
will result in incorrect translations. If the text is in Simplified Chinese use `zho-hans` for the
language code. If it is in Traditional Chinese, use `zho-hant`.

| ISO-639-2 | ISO-639-1 | Language             |
| --------- | --------- | -------------------- |
| alb       |  sq       | Albanian             |
| ara       |  ar       | Arabic               |
| aze       |  az       | Azerbaijani          |
| ben       |  bn       | Bengali              |
| bul       |  bg       | Bulgarian            |
| cat       |  ca       | Catalan              |
| cmn       |  zh       | Chinese Simplified\* |
| zho       |  zh       | Chinese Simplified\* |
| zho-hans  |  zh       | Chinese Simplified\* |
| zho-hant  |  zt       | Chinese Traditional\*|
| ces       |  cs       | Czech                |
| dan       |  da       | Danish               |
| nld       |  nl       | Dutch                |
| epo       |  eo       | Esperanto            |
| est       |  et       | Estonian             |
| fin       |  fi       | Finnish              |
| fra       |  fr       | French               |
| deu       |  de       | German               |
| ell       |  el       | Greek                |
| heb       |  he       | Hebrew               |
| hin       |  hi       | Hindi                |
| hun       |  hu       | Hungarian            |
| ind       |  id       | Indonesian           |
| gle       |  ga       | Irish                |
| ita       |  it       | Italian              |
| jpn       |  ja       | Japanese             |
| kor       |  ko       | Korean               |
| lav       |  lv       | Latvian              |
| lit       |  lt       | Lithuanian           |
| may       |  ms       | Malay \*\*           |
| msa       |  ms       | Malay \*\*           |
| nor       |  no       | Norwegian            |
| fas       |  fa       | Persian              |
| pol       |  pl       | Polish               |
| por       |  pt       | Portuguese           |
| rum       |  ro       | Romanian \*\*        |
| ron       |  ro       | Romanian \*\*        |
| rus       |  ru       | Russian              |
| srp       |  sr       | Serbian              |
| slk       |  sk       | Slovak \*\*          |
| slo       |  sk       | Slovak \*\*          |
| slv       |  sl       | Slovenian            |
| spa       |  es       | Spanish              |
| swe       |  sv       | Swedish              |
| tgl       |  tl       | Tagalog              |
| tha       |  th       | Thai                 |
| tur       |  tr       | Turkish              |
| ukr       |  uk       | Ukrainian            |

\* Technically two major scripts exist for modern Chinese (Traditional and Simplified). To differentiate between the scripts, ISO 15924 is provided as an addendum to the ISO 639-2 code.

\*\* Several languages like Malay, Romanian, and Slovak have two sets of ISO 639-2 codes.
