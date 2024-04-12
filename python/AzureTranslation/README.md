# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Translation Component. This component utilizes the [Azure Cognitive Services
Translator REST endpoint](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-translate)
to translate the content of detection properties. It has been tested against v3.0 of the
API.

This component translates the content of existing detection properties,
so it only makes sense to use it with
[feed forward](https://openmpf.github.io/docs/site/Feed-Forward-Guide) and
when it isn't the first element of a pipeline.

When a detection property is translated, the translation is put in to a new
detection property named `TRANSLATION`. The original detection property is not
modified. A property named `TRANSLATION TO LANGUAGE` containing the BCP-47
language code of the translated text will also be added. If the language
of the input text is detected to be the same as the `TO_LANGUAGE` job property,
then no translation will occur. When translation is skipped because of
matching languages, the `TRANSLATION` detection property will be omitted and
`SKIPPED TRANSLATION=TRUE` will be added to the detection properties.

When the source text is multiple languages, the translation endpoint will only
translate one of the languages. For example, translating
"你叫什么名字？ ¿Cómo te llamas?" to English results in
"What is your name? The Cómo te llamas?".


# Required Job Properties
In order for the component to process any jobs, the job properties listed below
must be provided. Neither has a default value.

- `ACS_URL`: Base URL for the Azure Cognitive Services Translator Endpoint.
<<<<<<< HEAD
   e.g. `https://api.cognitive.microsofttranslator.com` or
   `https://<custom-translate-host>/translator/text/v3.0`. The URL should
   not end with `/translate` because two separate endpoints are
   used. `ACS_URL + '/translate'` is used for translation.
   This property can also be configured
   using an environment variable named `MPF_PROP_ACS_URL`.
=======
  e.g. `https://api.cognitive.microsofttranslator.com` or
  `https://<custom-translate-host>/translator/text/v3.0`. The URL should
  not end with `/translate` because two separate endpoints are
  used. `ACS_URL + '/translate'` is used for translation.
  `ACS_URL + '/breaksentence'` is used to break up text when it is too long
  for a single translation request. This property can also be configured
  using an environment variable named `MPF_PROP_ACS_URL`.
>>>>>>> origin

- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an
  Azure Cognitive Services account. This property can also be configured
  using an environment variable named `MPF_PROP_ACS_SUBSCRIPTION_KEY`.


# Important Job Properties:
- `TO_LANGUAGE`: The BCP-47 language code for language that the properties
  should be translated to.


- `FEED_FORWARD_PROP_TO_PROCESS`: Comma-separated list of property names indicating
  which properties in the feed-forward track or detection to consider
  translating. For example, `TEXT,TRANSCRIPT`. If the first property listed is
  present, then that property  will be translated. If it's not, then the next
  property in the list is considered. At most, one property will be translated.

- `FROM_LANGUAGE`: In most cases, this property should not be used. It should
  only be used when automatic language detection is detecting the wrong
  language: Users can provide a BCP-47 code to force the translation service
  to translate text with a corrected source language.
  For instance, if incoming text is incorrectly being detected by Azure as
  Spanish instead of English, users can set `FROM_LANGUAGE=en`
  to force the service to treat all submitted text in the current job as English.

  Providing this property prevents the translation endpoint from
  doing automatic language detection. If `FROM_LANGUAGE` is provided, and the
  text is actually another language, the translation endpoint will return the
  input text unchanged.

- `SUGGESTED_FROM_LANGUAGE`: Optional property that indicates the fallback source
  BCP-47 language code to use when automatic language detection fails.
  The value from this property is only used when automatic language detection fails.
  `SUGGESTED_FROM_LANGUAGE` is the preferred setting to adjust when users know
  they are processing a large amount of text in a particular language, but other
  source languages may be present in individual pieces of text.
  For instance, setting `SUGGESTED_FROM_LANGUAGE=es` would allow the component to
  default to translating from Spanish, whenever Azure's language detector fails
  to identify the source language of the incoming text.


# Text Splitter Job Properties
The following settings control the behavior of dividing input text into acceptable chunks
for processing.

Through preliminary investigation, we identified the [WtP library ("Where's the
Point")](https://github.com/bminixhofer/wtpsplit) and spaCy's multilingual sentence
detection model for identifying sentence breaks in a large section of text.

WtP models are trained to split up multilingual text by sentence without the need of an
input language tag. The disadvantage is that the most accurate WtP models will need ~3.5
GB of GPU memory. On the other hand, spaCy has a single multilingual sentence detection
that appears to work better for splitting up English text in certain cases, unfortunately
this model lacks support handling for Chinese punctuation.

- `SENTENCE_MODEL`: Specifies the desired WtP or spaCy sentence detection model. For CPU
  and runtime considerations, the author of WtP recommends using `wtp-bert-mini`. More
  advanced WtP models that use GPU resources (up to ~8 GB) are also available. See list of
  WtP model names
  [here](https://github.com/bminixhofer/wtpsplit?tab=readme-ov-file#available-models). The
  only available spaCy model (for text with unknown language) is `xx_sent_ud_sm`.

  Review list of languages supported by WtP
  [here](https://github.com/bminixhofer/wtpsplit?tab=readme-ov-file#supported-languages).
  Review models and languages supported by spaCy [here](https://spacy.io/models).

- `SENTENCE_SPLITTER_CHAR_COUNT`: Specifies maximum number of characters to process
  through sentence/text splitter. Default to 500 characters as we only need to process a
  subsection of text to determine an appropriate split. (See discussion of potential char
  lengths
  [here](https://discourse.mozilla.org/t/proposal-sentences-lenght-limit-from-14-words-to-100-characters).

- `SENTENCE_SPLITTER_INCLUDE_INPUT_LANG`: Specifies whether to pass input language to
  sentence splitter algorithm. Currently, only WtP supports model threshold adjustments by
  input language.


# Listing Supported Languages
To list the supported languages replace `${ACS_URL}` and
`${ACS_SUBSCRIPTION_KEY}` in the following command and run it:

```shell script
curl -H "Ocp-Apim-Subscription-Key: ${ACS_SUBSCRIPTION_KEY}" "https://${ACS_URL}/languages?api-version=3.0&scope=translation"
```


# Sample Program
`sample_acs_translator.py` can be used to quickly test with the Azure
endpoint. It translates strings provided via command line arguments.
