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

- `FILTER_REASONABLE_LANGUAGES`: When set to false, attempt to return a langauge detection even if the langauge detector is not fully confident. It is recommend to disable this filter when using the OpenNLP language detector, as it has a strict confidence threshold.

Language results are stored as follows:
- `TEXT_LANGUAGE` : The primary detected language for a given text. Set to "Unknown" if no language is identified.
- `ISO_LANGUAGE` : The primary detected language for a given text in ISO 639-3 format. Set to "UNKNOWN" if no language id identified (as "UNK" is an ISO code).


# Depreciated Language Detectors:

- [Apache Tika's Language Detector](https://tika.apache.org/2.4.1/api/org/apache/tika/langdetect/tika/TikaLanguageDetector.html) (Old):

  This was Tika's original language detector, which "computes vector distance of trigrams between input string and language models".
  It is "not suitable for short texts" and is currently depreciated.

# Supported Language Detectors:



This component supports the following language detectors. Users can select their preferred detector using
the `LANGUAGE_DETECTOR` option:

- `LANGUAGE_DETECTOR = opennlp`: [Apache Tika OpenNLP Language Detector](https://opennlp.apache.org/)

  Apache Tika's latest in-house language detection capability based on OpenNLP's language detector.
  Uses Machine Learning (ML) models trained on the following datasets and supports 148 languages in total
    - [Leipzig corpus](https://wortschatz.uni-leipzig.de/en/download)
    - [cc-100](https://data.statmt.org/cc-100/)

  Supports almost every language in Optimaize except Aragonese.

  **PLEASE NOTE**, if `opennlp` is selected, please ensure that `FILTER_REASONABLE_LANGUAGES = FALSE`. OpenNLP has a stricter confidence threshold than Optimaize, although testing has found that it still selects the correct language for most cases.

- `LANGUAGE_DETECTOR = optimaize`: [Optimaize Language Detector](https://github.com/optimaize/language-detector)

  Third party language detection project that supports 71 languages.
  Predicts target language using N-gram frequency matching between input and language profiles.
  Supports almost every language present in Tika's Language Detector except Esperanto.
  Please note that Optimaize supports Punjabi/Panjabi while OpenNLP supports Western Punjabi/Panjabi.


# Supported Language List:

- Optimaize Language Detector:
  - **Aragonese** (Unique to this detector)
  - Afrikaans
  - Albanian
  - Arabic
  - Asturian
  - Basque
  - Belarusian
  - Bengali
  - Breton
  - Bulgarian
  - Catalan
  - Croatian
  - Czech
  - Danish
  - Dutch
  - English
  - Estonian
  - Finnish
  - French
  - Galician
  - German
  - Greek
  - Gujarati
  - Haitian
  - Hebrew
  - Hindi
  - Hungarian
  - Icelandic
  - Indonesian
  - Irish
  - Italian
  - Japanese
  - Kannada
  - Khmer
  - Korean
  - Latvian
  - Lithuanian
  - Macedonian
  - Malay
  - Malayalam
  - Maltese
  - Marathi
  - Nepali
  - Norwegian
  - Occitan
  - Persian
  - Polish
  - Portuguese
  - Punjabi
  - Romanian
  - Russian
  - Serbian
  - Simplified Chinese
  - Slovak
  - Slovakian
  - Slovene  - Esperanto
  - Slovenian
  - Somali
  - Spanish
  - Swahili
  - Swedish
  - Tagalog
  - Tamil
  - Telugu
  - Thai
  - Traditional Chinese
  - Turkish
  - Ukrainian
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