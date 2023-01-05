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

Lanuage packages are downloaded dynamically as needed. In addition, when building a Docker image the Dockerfile pre-installs German, French, Russian, and Spanish.

Note: Argos underperforms when translating to and from Chinese

| ISO-639-1 | ISO-639-2 | Language         |
| --- |---|------------------|
| `ar` | `ara` | Arabic           |
| `az` | `aze` | Azerbaijani      |
| `zh` | `cmn` | Chinese          |
| `cs` | `ces` | Czech            |
| `da` | `dan` | Danish           |
| `nl` | `nld` | Dutch            |
| `eo` | `epo` | Esperanto        |
| `fi` | `fin` | Finnish          |
| `fr` | `fra` | French           |
| `de` | `deu` | German           |
| `el` | `ell` | Greek            |
| `he` | `heb` | Hebrew           |
| `hi` | `hin` | Hindi            |
| `hu` | `hun` | Hungarian        |
| `id` | `ind` | Indonesian       |
| `ga` | `gle` | Irish            |
| `it` | `ita` | Italian          |
| `ja` | `jpn` | Japanese         |
| `ko` | `kor` | Korean           |
| `fa` | `fas` | Persian          |
| `pl` | `pol` | Polish           |
| `pt` | `por` | Portuguese       |
| `ru` | `rus` | Russian          |
| `sk` | `slk` | Slovak           |
| `es` | `spa` | Spanish          |
| `sv` | `swe` | Swedish          |
| `tr` | `tur` | Turkish          |
| `uk` | `ukr` | Ukrainian        |


