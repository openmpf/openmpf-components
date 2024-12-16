# Overview
This repository contains source code for the OpenMPF FastText Language Detection Component. This
component utilizes the [GlotLID](https://github.com/cisnlp/GlotLID) language identification model
and the [fastText](https://github.com/facebookresearch/fastText) library to perform language
identification on text.

# Primary Job Properties

- `FEED_FORWARD_PROP_TO_PROCESS`: Comma-separated list of property names indicating
  which properties in the feed-forward track or detection to perform language identification on.
  For example, `TEXT,TRANSCRIPT`. If the first property listed is present, then that property
  will be used. If it's not, then the next property in the list is considered. At most, one
  property will be processed.

# Output Detection Properties:

- If a language is detected, then the `ISO_LANGUAGE` and `ISO_SCRIPT` properties will
  be populated with the ISO-639-3 and ISO-15924 codes for language and script
  respectively. This represents the top predicted language for any given text
  input. If no language is detected, `ISO_LANGUAGE` and `ISO_SCRIPT` will be set to
  `<UNKNOWN>`.

- `PRIMARY_LANGUAGE_CONFIDENCE` contains the confidence that the `ISO_LANGUAGE` and `ISO_SCRIPT`
  properties are correct.


# Supported Languages

The component supports all of the [languages supported by the GlotLID model](https://github.com/cisnlp/GlotLID/blob/bd6bdf276013176d0ade3dfe33f5df590bf3eaca/languages-v3.md)
except for `cmn_Hani` and `lzh_Hani`. `cmn_Hani` will be reported as `zho_Hans` or `zho_Hant`.
`lzh_Hani` will never be reported.

The model does not differentiate between Simplified and Traditional Chinese. It reports `cmn_Hani`
for both. If the model outputs `cmn_Hani`, the component will either output `zho_Hans` or
`zho_Hant`. If the input text contains any characters that are only used in Traditional Chinese,
the component will report `zho_Hant`. `zho_Hans` is reported when the text contains Simplified
Characters and when the text only contains characters used in both Simplified and Traditional
Chinese.

The component will never output `lzh_Hani`. `lzh_Hani` refers to
[Classical Chinese](https://en.wikipedia.org/wiki/Classical_Chinese), an arcane form of Chinese
writing. When the model is provided with things like text with only whitespace or text with only
emoji, it will output `lzh_Hani` with a very high confidence.
