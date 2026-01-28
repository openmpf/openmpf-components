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
  e.g. `https://api.cognitive.microsofttranslator.com` or
  `https://<custom-translate-host>/translator/text/v3.0`. The URL should
  not end with `/translate` because two separate endpoints are
  used. `ACS_URL + '/translate'` is used for translation.
  This property can also be configured using an environment variable
  named `MPF_PROP_ACS_URL`.

- `ACS_SUBSCRIPTION_KEY`: A string containing your Azure Cognitive Services
  subscription key. To get one you will need to create an
  Azure Cognitive Services account. This property can also be configured
  using an environment variable named `MPF_PROP_ACS_SUBSCRIPTION_KEY`.


# Primary Job Properties
- `TO_LANGUAGE`: The BCP-47 language code for the language that the properties
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

- `ACS_SUBSCRIPTION_REGION`: Optional property that specifies the subscription
  region for the Azure Cognitive Services resource, such as 'eastus'. Required
  for some Azure deployments. If provided, will be set in the
  'Ocp-Apim-Subscription-Region' request header.


# Text Splitter Job Properties
The following settings control the behavior of dividing input text into acceptable chunks
for processing.

Through preliminary investigation, we identified the [SaT/WtP library ("Segment any Text" / "Where's the
Point")](https://github.com/bminixhofer/wtpsplit) and [spaCy's multilingual sentence
detection model](https://spacy.io/models) for identifying sentence breaks
in a large section of text.

SaT/WtP models are trained to split up multilingual text by sentence without the need of an
input language tag. The disadvantage is that the most accurate WtP models will need ~3.5
GB of GPU memory. SaT models are a more recent addition and considered to be a more accurate
set of sentence segmentation models; their resource costs are similar to WtP.

On the other hand, spaCy has a single multilingual sentence detection
that appears to work better for splitting up English text in certain cases, unfortunately
this model lacks support handling for Chinese punctuation.

- `SENTENCE_MODEL`: Specifies the desired SaT/WtP or spaCy sentence detection model. For CPU
  and runtime considerations, the authors of SaT/WtP recommends using `sat-3l-sm` or `wtp-bert-mini`.
  More advanced SaT/WtP models that use GPU resources (up to ~8 GB for WtP) are also available.

  See list of model names below:

  - [WtP Models](https://github.com/segment-any-text/wtpsplit/tree/1.3.0?tab=readme-ov-file#available-models)
  - [SaT Models](https://github.com/bminixhofer/wtpsplit?tab=readme-ov-file#available-models).

    Please note, the only available spaCy model (for text with unknown language) is `xx_sent_ud_sm`.

  Review list of languages supported by SaT/WtP below:

  - [WtP Models](https://github.com/segment-any-text/wtpsplit/tree/1.3.0?tab=readme-ov-file#supported-languages)
  - [SaT Models](https://github.com/bminixhofer/wtpsplit?tab=readme-ov-file#supported-languages)

  Review models and languages supported by spaCy [here](https://spacy.io/models).

- `SENTENCE_SPLITTER_CHAR_COUNT`: Specifies maximum number of characters to process
  through sentence/text splitter. Default to 500 characters as we only need to process a
  subsection of text to determine an appropriate split. (See discussion of potential char
  lengths
  [here](https://discourse.mozilla.org/t/proposal-sentences-lenght-limit-from-14-words-to-100-characters).

- `SENTENCE_SPLITTER_MODE`: Specifies text splitting behavior, options include:
  - `DEFAULT` : Splits text into chunks based on the `SENTENCE_SPLITTER_CHAR_COUNT` limit.
  - `SENTENCE`: Splits text at detected sentence boundaries. This mode creates more sentence breaks than `DEFAULT`, which is more focused on avoiding text splits unless the chunk size is reached.

- `SENTENCE_SPLITTER_INCLUDE_INPUT_LANG`: Specifies whether to pass input language to
  sentence splitter algorithm. Currently, only SaT/WtP supports model threshold adjustments by
  input language.

- `SENTENCE_MODEL_CPU_ONLY`: If set to TRUE, only use CPU resources for the sentence
  detection model. If set to FALSE, allow sentence model to also use GPU resources.
  For most runs using spaCy `xx_sent_ud_sm`, `sat-3l-sm`, or `wtp-bert-mini` models, GPU resources
  are not required. If using more advanced WtP models like `wtp-canine-s-12l`,
  it is recommended to set `SENTENCE_MODEL_CPU_ONLY=FALSE` to improve performance.
  That WtP model can use up to ~3.5 GB of GPU memory.

  Please note, to fully enable this option, you must also rebuild the Docker container
  with the following change: Within the Dockerfile, set `ARG BUILD_TYPE=gpu`.
  Otherwise, PyTorch will be installed without cuda support and
  component will always default to CPU processing.

- `SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE`: More advanced WTP models will
  require a target language. This property sets the default language to use for
  sentence splitting, and is overwritten whenever `FROM_LANGUAGE`, `SUGGESTED_FROM_LANGUAGE`,
  or Azure language detection return a different, WtP-supported language option.


# Listing Supported Languages
To list the supported languages replace `${ACS_URL}` and `${ACS_SUBSCRIPTION_KEY}` in the
following command and run it:
```shell script
curl -H "Ocp-Apim-Subscription-Key: ${ACS_SUBSCRIPTION_KEY}" "https://${ACS_URL}/languages?api-version=3.0&scope=translation"
```


# Sample Program
`sample_acs_translator.py` can be used to quickly test with the Azure
endpoint. It translates strings provided via command line arguments.
