# Overview

This directory contains source code for the OpenMPF Tika text detection component.

This component extracts text embedded in document formats, such as PDF (`.pdf`), PowerPoint (`.pptx`), Word (`.docx`),
Excel (`.xlsx`), OpenDocument Presentation (`.odp`), OpenDocument Text (`.odt`), OpenDocument Spreadsheet (`.ods`), and
text (`.txt`) documents. It also processes the text for detected languages
(71 languages currently supported).

PDF and PowerPoint document text will be extracted and processed per page / slide. Word, OpenDocument Text, and
OpenDocument Presentation documents cannot be parsed by page / slide. This component can still extract all of the text
from documents that cannot be parsed by page, but all of their tracks will be reported with `PAGE_NUM = 1`. Note that
page numbers start at 1, not 0, unless otherwise noted.

Every page can generate zero or more tracks, depending on the number of text sections in that page. A text section can
be a line or paragrah of text surrounded by newlines and/or page breaks, a single bullet point, a single table cell,
etc. In addition to `PAGE_NUM`, each track will also have a `SECTION_NUM` property. `SECTION_NUM` starts over at 1 on
each page / slide.

Users can also enable metadata reporting. If enabled by setting the job property `STORE_METADATA = true`, document
metadata will be labeled and stored as the first track. Metadata track will not contain the `PAGE_NUM`, `SECTION_NUM`,
or `TEXT` detection properties. Instead, the track will have a `METADATA` property with a value formatted as a JSON
string. The value can contain multiple subfields, depending on the document format, like `Content-Encoding`,
`Content-Type`, etc. For example:

```
METADATA = {"X-Parsed-By":"org.apache.tika.parser.DefaultParser","Content-Encoding":"ISO-8859-1","Content-Type":"text/plain; charset=ISO-8859-1"}
```

# Format-Specific Behaviors

The following format-specific behaviors were observed using Tika 1.28.1 on Ubuntu 20.04:

- For all formats except PDF and PowerPoint documents, we intentionally set `PAGE_NUM = -1` to indicate that the page
  number cannot be determined.

- For `.txt` files and Excel documents, all of the text will be reported in a single track with `PAGE_NUM = -1`
  and `SECTION_NUM = 1`.

- OpenDocument Spreadsheet documents will generate one track per cell, as well as some additional tracks with
  date and time information, "Page /", and "???".

# Language Detection Parameters

Tika supports the following language detection properties:

- `MIN_LANGUAGES`: When set to a positive integer, attempt to return specified number of top languages, even if some are not marked as reasonably certain. Non-positive values disable this property to only accept reasonable predictions.

For instance, if `MIN_LANGUAGES` is set to 2, the component will always attempt to return the top 2 predicted languages, followed by the next languages if they are also marked as reasonably certain.

If `MAX_REASONABLE_LANGUAGES` is set to -1 and `MIN_LANGUAGES` is set to 2 (default), the component will always attempt to return the top predicted language and a secondary language, even if they are not set to reasonably confident by Tika.

Please note that the behavior of `MIN_LANGUAGES` is different depending on the language detector:
- The Optimaize language detector will often produce the exact number of `MIN_LANGUAGES` requested by a user.
- The OpenNLP language detector tends to only produce 1 language unless `MIN_LANGUAGES` is set to 2 or greater. However, low confidence results are still sometimes thrown out even if the user requested additional language predictions. (This indicates that OpenNLP uses a different filtering mechanism which could be investigated further).

Language results are stored as follows:
- `TEXT_LANGUAGE` : The primary detected language for a given text. Set to "Unknown" if no language is identified.
- `TEXT_LANGUAGE_CONFIDENCE` : Raw confidence score for the primary language. Ranges from 0 to 1 for OpenNLP and Optimaize language detectors, and set to -1 if no results are found. See [note here.](https://tika.apache.org/2.4.1/api/index.html?org/apache/tika/language/detect/LanguageResult.html)
- `ISO_LANGUAGE` : The primary detected language for a given text in ISO 639-3 format. Set to "UNKNOWN" if no language id identified (as "UNK" is an ISO code).

Secondary languages and their confidence scores are listed as comma separated strings:
- `SECONDARY_TEXT_LANGUAGES` : A list of secondary languages (from greatest to least confidence) separated by ", " delimiters.
    Note,`SECONDARY_TEXT` properties are not included if no secondary languages are identified.
- `SECONDARY_TEXT_LANGUAGE_CONFIDENCES` : A confidence list corresponding to the secondary detected languages in order (also separated by commas). 

  For secondary language predictions, ensure that `MIN_LANGUAGES` property is set to 2 or greater. Secondary language results are typically ignored by the component as Tika has a high confidence threshold for reasonable predictions.

# Supported Language Detectors:

This component supports the following language detectors. Users can select their preferred detector using
the `LANG_DETECTOR` option:
                                                                                                                                                                          
- `LANG_DETECTOR = opennlp`: [Apache Tika OpenNLP Language Detector](https://tika.apache.org/2.4.1/api/org/apache/tika/langdetect/opennlp/OpenNLPDetector.html)           
                                                                                                                                                                          
  Apache Tika's latest in-house language detection capability based on OpenNLP's language detector.                                                                       
  Uses Machine Learning (ML) models trained on the following datasets and supports 148 languages in total                                                                   
    - [Leipzig corpus](https://wortschatz.uni-leipzig.de/en/download)                                                                                                     
    - [cc-100](https://data.statmt.org/cc-100/)                                                                                                                           
                                                                                                                                                                          
  Supports almost every language in Optimaize and Tika Language Detectors except Aragonese.  

  **PLEASE NOTE**, if `opennlp` is selected, ensure that the `MIN_LANGUAGES` property is set to 1 or greater.

- `LANG_DETECTOR = optimaize`: [Optimaize Language Detector](https://github.com/optimaize/language-detector)

  Third party language detection project that supports 71 languages.
  Predicts target language using N-gram frequency matching between input and language profiles.
  Supports almost every language present in Tika's Language Detector except Esperanto.
  Please note that Optimaize supports Punjabi/Panjabi while OpenNLP supports Western Punjabi/Panjabi.
 
   
# Supported Language List:

- Tika Language Detector:
  - Belarusian
  - Catalan
  - Danish
  - Dutch
  - English
  - Esperanto
  - Estonian
  - Finnish
  - French
  - Galician
  - German
  - Greek
  - Hungarian
  - Icelandic
  - Italian
  - Lithuanian
  - Norwegian
  - Persian
  - Polish
  - Portuguese
  - Romanian
  - Russian
  - Slovakian
  - Slovenian
  - Spanish
  - Swedish
  - Thai
  - Ukrainian


- Optimaize Language Detector:
  - Every language in Tika's language detector, except Esperanto.
  - Afrikaans
  - **Aragonese** (Unique to this detector)
  - Albanian
  - Arabic
  - Asturian
  - Basque
  - Bengali
  - Breton
  - Bulgarian
  - Croatian
  - Czech
  - Gujarati
  - Haitian
  - Hebrew
  - Hindi
  - Indonesian
  - Irish
  - Japanese
  - Kannada
  - Khmer
  - Korean
  - Latvian
  - Macedonian
  - Malay
  - Malayalam
  - Maltese
  - Marathi
  - Nepali
  - Occitan
  - Punjabi
  - Serbian
  - Simplified Chinese
  - Slovak
  - Slovene
  - Somali
  - Swahili
  - Tagalog
  - Tamil
  - Telugu
  - Traditional Chinese
  - Turkish
  - Urdu
  - Vietnamese
  - Walloon
  - Welsh
  - Yiddish
                        

- OpenNLP Language Detector:
  - Every language in Tika and Optimaize Language Detectors except Aragonese. 
  - Armenian
  - Assamese
  - Azerbaijani
  - Balinese
  - Bashkir
  - Bihari languages
  - Burmese
  - Cebuano
  - Dhivehi
  - Eastern Mari
  - Esperanto
  - Faroese
  - Fulah
  - Goan Konkani
  - Igbo
  - Iranian Persian
  - Javanese
  - Konkani
  - Kurdish
  - Lingala
  - Low German
  - Malay
  - Mandarin Chinese
  - Maori
  - Min Nan Chinese
  - Minangkabau
  - Mongolian
  - Paraguayan Guaraní
  - Pedi
  - Pushto
  - Scottish Gaelic
  - Sindhi
  - Slovenian
  - Standard Estonian
  - Standard Latvian
  - Swati
  - Swiss German
  - Tajik
  - Turkmen
  - Uighur
  - Uzbek
  - Western Frisian
  - Western Panjabi
  - Xhosa
  - Yoruba
  - Zulu