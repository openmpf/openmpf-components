# Overview

This repository contains source code for the OpenMPF Azure Cognitive Services
Translation Component. This component utilizes the [Azure Cognitive Services 
Translator REST endpoint](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-translate)
to translate the content of detection properties. It has only been tested
against v3.0 of the API.

This component translates the content of existing detection properties,
so it only makes sense to use it with 
[feed forward](https://openmpf.github.io/docs/site/Feed-Forward-Guide) and 
when it isn't the first element of a pipeline.
 
When a detection property is translated, the translation is put in to a new 
detection property named `TRANSLATION`. The original detection property is not 
modified. A property named `TRANSLATION TO LANGUAGE` containing the BCP-47 
language code of the translated text will also be added.

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
   `ACS_URL + '/breaksentence'` is used to break up text when it is too long
   for a single translation request. This property can also be configured
   using an environment variable named `MPF_PROP_ACS_URL`.
   
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
  language. Providing this property prevents the translation endpoint from 
  doing automatic language detection. If `FROM_LANGUAGE` is provided, and the 
  text is actually another language, the translation endpoint will return the 
  input text unchanged.
  

# Listing Supported Languages
To list the supported languages replace `${ACS_URL}` and 
`${ACS_SUBSCRIPTION_KEY}` in the following command and run it:
```shell script
curl -H "Ocp-Apim-Subscription-Key: ${ACS_SUBSCRIPTION_KEY}" "https://${ACS_URL}/languages?api-version=3.0&scope=translation"
```


# Sample Program
`sample_acs_translator.py` can be used to quickly test with the Azure
endpoint. It translates strings provided via command line arguments.