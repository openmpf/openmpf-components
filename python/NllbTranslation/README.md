# Overview

This repository contains source code for the OpenMPF No Language Left Behind component and is based on [Meta's No Language Left Behind Project](https://ai.meta.com/research/no-language-left-behind/). The component translates input text from a given source language to English. The source language can be provided as a job property, or be indicated in the detection properties from a feed-forward track.

By default, this component is configured to use the **`facebook/nllb-200-3.3B` model**, the largest of Meta’s [No Language Left Behind (NLLB)](https://huggingface.co/models?search=facebook/nllb) models. This provides the highest translation quality, but also requires significant hardware resources.

To accommodate smaller deployment enviroments, this component can use smaller NLLB models, such as [nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) or [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M).

# Recommended System Requirements

- **GPU (recommended for default 3.3B model)**
  - NVIDIA GPU with CUDA support
  - At least **24 GB of GPU VRAM**

- **CPU-only (not recommended for 3.3B model unless sufficient memory is available)**
  - At least **32 GB of system RAM**

### Example Model Requirements

| Model Name                              | Parameters | Minimum RAM (CPU) | Minimum GPU VRAM |
|-----------------------------------------|------------|-------------------|------------------|
| `facebook/nllb-200-3.3B` (default)      | 3.3B       | 32 GB             | 24 GB            |
| `facebook/nllb-200-1.3B`                | 1.3B       | 16 GB             | 12 GB            |
| `facebook/nllb-200-distilled-1.3B`      | 1.3B       | 16 GB             | 12 GB            |
| `facebook/nllb-200-distilled-600M`      | 0.6B       | 8 GB              | 8 GB             |

> ⚠️ **Note**: The listed memory requirements are approximate and may vary depending on your environment, framework version, and batch size. For production deployments, allocate additional memory beyond the minimums.

# Job Properties
The below properties can be optionally provided to alter the behavior of the component.

- `FEED_FORWARD_PROP_TO_PROCESS`: Controls which properties of the feed-forward track or detection are considered for translation. This should be a comma-separated list of property names. The default properties are `"TEXT,TRANSCRIPT"`. If `TRANSLATE_ALL_FF_PROPERTIES` is set to TRUE, then each named property in the list that is present is translated and the resulting translations will be added to the feed forward track. The default properties will be saved as `TEXT TRANSLATION` or `TRANSCRIPT TRANSLATION` depending on which (or both) is present. For other property names that are provided, the translation property will follow the same convention and translations will be saved to a `[PROPERTY NAME] TRANSLATION`.

- `TRANSLATE_ALL_FF_PROPERTIES`: If set to TRUE, translate all properties of the feed-forward track or detection listed in `FEED_FORWARD_PROP_TO_PROCESS`. When set to FALSE, only translate the first property, and the translation is stored in the `TRANSLATION` output property.

- `LANGUAGE_FEED_FORWARD_PROP`: Is an optional feed-forward property that comes from an earlier stage in a pipeline, indicating which language to translate from. The default values are `"ISO_LANGUAGE, DECODED_LANGUAGE, LANGUAGE"`. The ISO 639-3 standard is the expected input for any of these fields. If the first property listed is present, then that property will be used. If it's not, then the next property in the list is considered. If none are present, the language to translate from will be determined by the DEFAULT_SOURCE_LANGUAGE property instead.

- `SCRIPT_FEED_FORWARD_PROP`: Is an optional feed-forward property that is set from an earlier stage in the a pipeline, indicating which source script to use during translation. The default values are `"ISO_SCRIPT, DECODED_SCRIPT, SCRIPT"`. The ISO 15924 is expected as the input for these fields. The first property that is found in the list will be used, otherwise the the DEFAULT_SOURCE_SCRIPT property will be used to select the script to be used for translation.

- `DEFAULT_SOURCE_LANGUAGE`: The default source language to use if none of the property names listed in `LANGUAGE_FEED_FORWARD_PROP` are present in a feed-forward track or detection. Values should be in the ISO 639-3 standard.

- `DEFAULT_SOURCE_SCRIPT`: The default source script to use if none of the property names listed in the `SCRIPT_FEED_FORWARD_PROP` are present in a feed-forward track or detection. Values should be in the ISO 15924 standard.

- `TARGET_LANGUAGE`: Optional property to define a language to translate to. This value defaults to english if the property is not present.

- `TARGET_SCRIPT`: Optional property to define a script to be used in translating a language to. This value defaults to latin.

- `NLLB_MODEL`: Specifies which No Language Left Behind (NLLB) model to use. The default model is `facebook/nllb-200-3.3B` and is included in the pre-built NLLB Translation docker image. If this property is configured with a different model, the component will attempt to download the specified model from Hugging Face. See [Recommended System Requirements](#recommended-system-requirements) for additional information.

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
  through sentence/text splitter. Default to 360 characters as we only need to process a
  subsection of text to determine an appropriate split. (See discussion of potential char
  lengths
  [here](https://discourse.mozilla.org/t/proposal-sentences-lenght-limit-from-14-words-to-100-characters)).

- `USE_NLLB_TOKEN_LENGTH`: When set to `TRUE`, the component measures input size in tokens (as produced by the
  currently-loaded NLLB model tokenizer) instead of characters.
  Set to `FALSE` to switch to the character-count limit specified by `SENTENCE_SPLITTER_CHAR_COUNT`.

- `NLLB_TRANSLATION_TOKEN_LIMIT`: Specifies the maximum number of tokens allowed per chunk before text is split.
  This property is only used when `USE_NLLB_TOKEN_LENGTH` is set to `True` and effectively replaces
  `SENTENCE_SPLITTER_CHAR_COUNT` when active.

- `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT`: Specifies the preferred (soft) token limit for translation chunks when `USE_NLLB_TOKEN_LENGTH=TRUE`.
  - If set to a value greater than 0 and less than or equal to `NLLB_TRANSLATION_TOKEN_LIMIT`, the text splitter will attempt to create chunks near this size.
  - When enabled, the splitter may split text even if the full text is under the hard token limit.
  - Slightly exceeding the soft limit is allowed when aligning to sentence boundaries.
  - Must be less than or equal to `NLLB_TRANSLATION_TOKEN_LIMIT`.
  - Default: `130` tokens estimated from experiments on text translation chunks.


  Based on the current models available:
  - https://huggingface.co/facebook/nllb-200-3.3B
  - https://huggingface.co/facebook/nllb-200-1.3B
  - https://huggingface.co/facebook/nllb-200-distilled-1.3B
  - https://huggingface.co/facebook/nllb-200-distilled-600M

  - The recommended token limit is 512 tokens, across all four NLLB models.

- `SENTENCE_SPLITTER_INCLUDE_INPUT_LANG`: Specifies whether to pass input language to
  sentence splitter algorithm. Currently, only WtP supports model threshold adjustments by
  input language.

- `SENTENCE_SPLITTER_MODE`: Specifies text splitting behavior, options include:
  - `DEFAULT` : Splits text into chunks based on the `SENTENCE_SPLITTER_CHAR_COUNT` limit.
  - `SENTENCE`: Splits text at detected sentence boundaries. This mode creates more sentence breaks than `DEFAULT`, which is more focused on avoiding text splits unless the chunk size is reached.
  - So far, experimentation suggests that `SENTENCE` splitting creates the risk of translation chunks that contain too few samples of text to properly generate an accurate translation. Thus, we recommend continuing with `DEFAULT` for most translation needs.

- `PROCESS_DIFFICULT_LANGUAGES`: Comma-separated list of languages that should use `DIFFICULT_LANGUAGE_TOKEN_LIMIT` as the preferred token chunk size during translation.
  - Default: `"arabic"`
  - Matching applies to ISO-639-3 codes (e.g., `arb`, `arz`) or language names such as `"arabic"`.
  - When active, the  `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` is replaced by a more aggressive `DIFFICULT_LANGUAGE_TOKEN_LIMIT`.

- `DIFFICULT_LANGUAGE_TOKEN_LIMIT`: Preferred number of tokens per chunk involving languages named in `PROCESS_DIFFICULT_LANGUAGES`. When active, this overrides `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` but does not alter `NLLB_TRANSLATION_TOKEN_LIMIT`.
  - Only used when `USE_NLLB_TOKEN_LENGTH=TRUE`.
  - Overrides `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` for difficult languages.
  - Must be less than or equal to `NLLB_TRANSLATION_TOKEN_LIMIT`.
  - Default: `50` tokens.



- `SENTENCE_SPLITTER_NEWLINE_BEHAVIOR`: Specifies how individual newlines between characters should be handled when splitting text. Options include:
  - `GUESS` (default): Automatically replace newlines with either spaces or remove them, depending on the detected script between newlines.
  - `SPACE`: Always replaces newlines with a space, regardless of script.
  - `REMOVE`: Always removes newlines entirely, joining the adjacent characters directly.
  - `NONE`: Leaves newlines as-is in the input text.
  Please note that multiple adjacent newlines are treated as a manual text divide, across all settings. This is to ensure subtitles and other singular text examples are properly separated from other text during translation.

- `SENTENCE_MODEL_CPU_ONLY`: If set to TRUE, only use CPU resources for the sentence
  detection model. If set to FALSE, allow sentence model to also use GPU resources.
  For most runs using spaCy `xx_sent_ud_sm` or `wtp-bert-mini` models, GPU resources
  are not required. If using more advanced WtP models like `wtp-canine-s-12l`,
  it is recommended to set `SENTENCE_MODEL_CPU_ONLY=FALSE` to improve performance.
  That model can use up to ~3.5 GB of GPU memory.

  Please note, to fully enable this option, you must also rebuild the Docker container
  with the following change: Within the Dockerfile, set `ARG BUILD_TYPE=gpu`.
  Otherwise, PyTorch will be installed without cuda support and
  component will always default to CPU processing.

- `SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE`: More advanced WtP models require a language code.
  This property sets the default language to use for sentence splitting if no source language is available from
  `LANGUAGE_FEED_FORWARD_PROP` or `DEFAULT_SOURCE_LANGUAGE`.

   ## The following BCP language codes can be provided for WtP and SaT Text Splitters:
| iso | Name            | ISO-639-3 code accepted by conversion | Other accepted aliases   |
| --- | --------------- | ------------------------------------- | ------------------------ |
| af  | Afrikaans       | afr                                   |                          |
| am  | Amharic         | amh                                   |                          |
| ar  | Arabic          | ara                                   |                          |
| az  | Azerbaijani     | aze                                   |                          |
| be  | Belarusian      | bel                                   |                          |
| bg  | Bulgarian       | bul                                   |                          |
| bn  | Bengali         | ben                                   |                          |
| ca  | Catalan         | cat                                   | valencian                |
| ceb | Cebuano         | ceb                                   |                          |
| cs  | Czech           | ces                                   | cze                      |
| cy  | Welsh           | cym                                   | wel                      |
| da  | Danish          | dan                                   |                          |
| de  | German          | deu                                   | ger                      |
| el  | Greek           | ell                                   | gre                      |
| en  | English         | eng                                   |                          |
| eo  | Esperanto       | epo                                   |                          |
| es  | Spanish         | spa                                   | castilian                |
| et  | Estonian        | est                                   |                          |
| eu  | Basque          | eus                                   | baq                      |
| fa  | Persian         | fas                                   | per                      |
| fi  | Finnish         | fin                                   |                          |
| fr  | French          | fra                                   | fre                      |
| fy  | Western Frisian | fry                                   |                          |
| ga  | Irish           | gle                                   |                          |
| gd  | Scottish Gaelic | gla                                   | gaelic                   |
| gl  | Galician        | glg                                   |                          |
| gu  | Gujarati        | guj                                   |                          |
| ha  | Hausa           | hau                                   |                          |
| he  | Hebrew          | heb                                   |                          |
| hi  | Hindi           | hin                                   |                          |
| hu  | Hungarian       | hun                                   |                          |
| hy  | Armenian        | hye                                   | arm                      |
| id  | Indonesian      | ind                                   |                          |
| ig  | Igbo            | ibo                                   |                          |
| is  | Icelandic       | isl                                   | ice                      |
| it  | Italian         | ita                                   |                          |
| ja  | Japanese        | jpn                                   |                          |
| jv  | Javanese        | jav                                   |                          |
| ka  | Georgian        | kat                                   | geo                      |
| kk  | Kazakh          | kaz                                   |                          |
| km  | Central Khmer   | khm                                   |                          |
| kn  | Kannada         | kan                                   |                          |
| ko  | Korean          | kor                                   |                          |
| ku  | Kurdish         | kur                                   |                          |
| ky  | Kirghiz         | kir                                   | kyrgyz                   |
| la  | Latin           | lat                                   |                          |
| lt  | Lithuanian      | lit                                   |                          |
| lv  | Latvian         | lav                                   |                          |
| mg  | Malagasy        | mlg                                   |                          |
| mk  | Macedonian      | mkd                                   | mac                      |
| ml  | Malayalam       | mal                                   |                          |
| mn  | Mongolian       | mon                                   |                          |
| mr  | Marathi         | mar                                   |                          |
| ms  | Malay           | msa                                   | may                      |
| mt  | Maltese         | mlt                                   |                          |
| my  | Burmese         | mya                                   | bur                      |
| ne  | Nepali          | nep                                   |                          |
| nl  | Dutch           | nld                                   | dut, flemish             |
| no  | Norwegian       | nor                                   |                          |
| pa  | Panjabi         | pan                                   | punjabi                  |
| pl  | Polish          | pol                                   |                          |
| ps  | Pushto          | pus                                   | pashto                   |
| pt  | Portuguese      | por                                   |                          |
| ro  | Romanian        | ron                                   | rum, moldavian, moldovan |
| ru  | Russian         | rus                                   |                          |
| si  | Sinhala         | sin                                   | sinhalese                |
| sk  | Slovak          | slk                                   | slo                      |
| sl  | Slovenian       | slv                                   |                          |
| sq  | Albanian        | sqi                                   | alb                      |
| sr  | Serbian         | srp                                   |                          |
| sv  | Swedish         | swe                                   |                          |
| ta  | Tamil           | tam                                   |                          |
| te  | Telugu          | tel                                   |                          |
| tg  | Tajik           | tgk                                   |                          |
| th  | Thai            | tha                                   |                          |
| tr  | Turkish         | tur                                   |                          |
| uk  | Ukrainian       | ukr                                   |                          |
| ur  | Urdu            | urd                                   |                          |
| uz  | Uzbek           | uzb                                   |                          |
| vi  | Vietnamese      | vie                                   |                          |
| xh  | Xhosa           | xho                                   |                          |
| yi  | Yiddish         | yid                                   |                          |
| yo  | Yoruba          | yor                                   |                          |
| zh  | Chinese         | zho, cmn                              | chi, hans, hant          |
| zu  | Zulu            | zul                                   |                          |

# Supported NLLB Language Codes
The following are the ISO 639-3 and ISO 15924 codes, and their corresponding languages which NLLB can translate from.

| ISO-639-3 | ISO-15924  |              Language
| --------- | ---------- | ----------------------------------
|    ace    |    Arab    |   Acehnese Arabic
|    ace    |    Latn    |   Acehnese Latin
|    acm    |    Arab    |   Mesopotamian Arabic
|    acq    |    Arab    |   Ta’izzi-Adeni Arabic
|    aeb    |    Arab    |   Tunisian Arabic
|    afr    |    Latn    |   Afrikaans
|    ajp    |    Arab    |   South Levantine Arabic
|    aka    |    Latn    |   Akan
|    amh    |    Ethi    |   Amharic
|    apc    |    Arab    |   North Levantine Arabic
|    arb    |    Arab    |   Modern Standard Arabic
|    arb    |    Latn    |   Modern Standard Arabic (Romanized)
|    ars    |    Arab    |   Najdi Arabic
|    ary    |    Arab    |   Moroccan Arabic
|    arz    |    Arab    |   Egyptian Arabic
|    asm    |    Beng    |   Assamese
|    ast    |    Latn    |   Asturian
|    awa    |    Deva    |   Awadhi
|    ayr    |    Latn    |   Central Aymara
|    azb    |    Arab    |   South Azerbaijani
|    azj    |    Latn    |   North Azerbaijani
|    bak    |    Cyrl    |   Bashkir
|    bam    |    Latn    |   Bambara
|    ban    |    Latn    |   Balinese
|    bel    |    Cyrl    |   Belarusian
|    bem    |    Latn    |   Bemba
|    ben    |    Beng    |   Bengali
|    bho    |    Deva    |   Bhojpuri
|    bjn    |    Arab    |   Banjar (Arabic script)
|    bjn    |    Latn    |   Banjar (Latin script)
|    bod    |    Tibt    |   Standard Tibetan
|    bos    |    Latn    |   Bosnian
|    bug    |    Latn    |   Buginese
|    bul    |    Cyrl    |   Bulgarian
|    cat    |    Latn    |   Catalan
|    ceb    |    Latn    |   Cebuano
|    ces    |    Latn    |   Czech
|    cjk    |    Latn    |   Chokwe
|    ckb    |    Arab    |   Central Kurdish
|    crh    |    Latn    |   Crimean Tatar
|    cym    |    Latn    |   Welsh
|    dan    |    Latn    |   Danish
|    deu    |    Latn    |   German
|    dik    |    Latn    |   Southwestern Dinka
|    dyu    |    Latn    |   Dyula
|    dzo    |    Tibt    |   Dzongkha
|    ell    |    Grek    |   Greek
|    eng    |    Latn    |   English
|    epo    |    Latn    |   Esperanto
|    est    |    Latn    |   Estonian
|    eus    |    Latn    |   Basque
|    ewe    |    Latn    |   Ewe
|    fao    |    Latn    |   Faroese
|    fij    |    Latn    |   Fijian
|    fin    |    Latn    |   Finnish
|    fon    |    Latn    |   Fon
|    fra    |    Latn    |   French
|    fur    |    Latn    |   Friulian
|    fuv    |    Latn    |   Nigerian Fulfulde
|    gla    |    Latn    |   Scottish Gaelic
|    gle    |    Latn    |   Irish
|    glg    |    Latn    |   Galician
|    grn    |    Latn    |   Guarani
|    guj    |    Gujr    |   Gujarati
|    hat    |    Latn    |   Haitian Creole
|    hau    |    Latn    |   Hausa
|    heb    |    Hebr    |   Hebrew
|    hin    |    Deva    |   Hindi
|    hne    |    Deva    |   Chhattisgarhi
|    hrv    |    Latn    |   Croatian
|    hun    |    Latn    |   Hungarian
|    hye    |    Armn    |   Armenian
|    ibo    |    Latn    |   Igbo
|    ilo    |    Latn    |   Ilocano
|    ind    |    Latn    |   Indonesian
|    isl    |    Latn    |   Icelandic
|    ita    |    Latn    |   Italian
|    jav    |    Latn    |   Javanese
|    jpn    |    Jpan    |   Japanese
|    kab    |    Latn    |   Kabyle
|    kac    |    Latn    |   Jingpho
|    kam    |    Latn    |   Kamba
|    kan    |    Knda    |   Kannada
|    kas    |    Arab    |   Kashmiri (Arabic script)
|    kas    |    Deva    |   Kashmiri (Devanagari script)
|    kat    |    Geor    |   Georgian
|    knc    |    Arab    |   Central Kanuri (Arabic script)
|    knc    |    Latn    |   Central Kanuri (Latin script)
|    kaz    |    Cyrl    |   Kazakh
|    kbp    |    Latn    |   Kabiyè
|    kea    |    Latn    |   Kabuverdianu
|    khm    |    Khmr    |   Khmer
|    kik    |    Latn    |   Kikuyu
|    kin    |    Latn    |   Kinyarwanda
|    kir    |    Cyrl    |   Kyrgyz
|    kmb    |    Latn    |   Kimbundu
|    kmr    |    Latn    |   Northern Kurdish
|    kon    |    Latn    |   Kikongo
|    kor    |    Hang    |   Korean
|    lao    |    Laoo    |   Lao
|    lij    |    Latn    |   Ligurian
|    lim    |    Latn    |   Limburgish
|    lin    |    Latn    |   Lingala
|    lit    |    Latn    |   Lithuanian
|    lmo    |    Latn    |   Lombard
|    ltg    |    Latn    |   Latgalian
|    ltz    |    Latn    |   Luxembourgish
|    lua    |    Latn    |   Luba-Kasai
|    lug    |    Latn    |   Ganda
|    luo    |    Latn    |   Luo
|    lus    |    Latn    |   Mizo
|    lvs    |    Latn    |   Standard Latvian
|    mag    |    Deva    |   Magahi
|    mai    |    Deva    |   Maithili
|    mal    |    Mlym    |   Malayalam
|    mar    |    Deva    |   Marathi
|    min    |    Arab    |   Minangkabau (Arabic script)
|    min    |    Latn    |   Minangkabau (Latin script)
|    mkd    |    Cyrl    |   Macedonian
|    plt    |    Latn    |   Plateau Malagasy
|    mlt    |    Latn    |   Maltese
|    mni    |    Beng    |   Meitei (Bengali script)
|    khk    |    Cyrl    |   Halh Mongolian
|    mos    |    Latn    |   Mossi
|    mri    |    Latn    |   Maori
|    mya    |    Mymr    |   Burmese
|    nld    |    Latn    |   Dutch
|    nno    |    Latn    |   Norwegian Nynorsk
|    nob    |    Latn    |   Norwegian Bokmål
|    npi    |    Deva    |   Nepali
|    nso    |    Latn    |   Northern Sotho
|    nus    |    Latn    |   Nuer
|    nya    |    Latn    |   Nyanja
|    oci    |    Latn    |   Occitan
|    gaz    |    Latn    |   West Central Oromo
|    ory    |    Orya    |   Odia
|    pag    |    Latn    |   Pangasinan
|    pan    |    Guru    |   Eastern Panjabi
|    pap    |    Latn    |   Papiamento
|    pes    |    Arab    |   Western Persian
|    pol    |    Latn    |   Polish
|    por    |    Latn    |   Portuguese
|    prs    |    Arab    |   Dari
|    pbt    |    Arab    |   Southern Pashto
|    quy    |    Latn    |   Ayacucho Quechua
|    ron    |    Latn    |   Romanian
|    run    |    Latn    |   Rundi
|    rus    |    Cyrl    |   Russian
|    sag    |    Latn    |   Sango
|    san    |    Deva    |   Sanskrit
|    sat    |    Olck    |   Santali
|    scn    |    Latn    |   Sicilian
|    shn    |    Mymr    |   Shan
|    sin    |    Sinh    |   Sinhala
|    slk    |    Latn    |   Slovak
|    slv    |    Latn    |   Slovenian
|    smo    |    Latn    |   Samoan
|    sna    |    Latn    |   Shona
|    snd    |    Arab    |   Sindhi
|    som    |    Latn    |   Somali
|    sot    |    Latn    |   Southern Sotho
|    spa    |    Latn    |   Spanish
|    als    |    Latn    |   Tosk Albanian
|    srd    |    Latn    |   Sardinian
|    srp    |    Cyrl    |   Serbian
|    ssw    |    Latn    |   Swati
|    sun    |    Latn    |   Sundanese
|    swe    |    Latn    |   Swedish
|    swh    |    Latn    |   Swahili
|    szl    |    Latn    |   Silesian
|    tam    |    Taml    |   Tamil
|    tat    |    Cyrl    |   Tatar
|    tel    |    Telu    |   Telugu
|    tgk    |    Cyrl    |   Tajik
|    tgl    |    Latn    |   Tagalog
|    tha    |    Thai    |   Thai
|    tir    |    Ethi    |   Tigrinya
|    taq    |    Latn    |   Tamasheq (Latin script)
|    taq    |    Tfng    |   Tamasheq (Tifinagh script)
|    tpi    |    Latn    |   Tok Pisin
|    tsn    |    Latn    |   Tswana
|    tso    |    Latn    |   Tsonga
|    tuk    |    Latn    |   Turkmen
|    tum    |    Latn    |   Tumbuka
|    tur    |    Latn    |   Turkish
|    twi    |    Latn    |   Twi
|    tzm    |    Tfng    |   Central Atlas Tamazight
|    uig    |    Arab    |   Uyghur
|    ukr    |    Cyrl    |   Ukrainian
|    umb    |    Latn    |   Umbundu
|    urd    |    Arab    |   Urdu
|    uzn    |    Latn    |   Northern Uzbek
|    vec    |    Latn    |   Venetian
|    vie    |    Latn    |   Vietnamese
|    war    |    Latn    |   Waray
|    wol    |    Latn    |   Wolof
|    xho    |    Latn    |   Xhosa
|    ydd    |    Hebr    |   Eastern Yiddish
|    yor    |    Latn    |   Yoruba
|    yue    |    Hant    |   Yue Chinese
|    zho    |    Hans    |   Chinese (Simplified)
|    zho    |    Hant    |   Chinese (Traditional)
|    zsm    |    Latn    |   Standard Malay
|    zul    |    Latn    |   Zulu


## Analysis of NLLB Results
From `NLLB Token Length Investigation.xlsx` the team investigated how the NLLB translation capability handles text translations for small to very large chunks of text.

Overall our findings are:

1. Most languages have a breaking point occurring around 120-140 tokens. Translations after this point start to lose or forget parts of the text being translated.

2. Likewise, most languages also benefit from additional context clues and sections of text being submitted together. This helps provide sufficient translation context clues.

   - For instance, the term `Dracula` was actually mistranslated in one test when the word was presented on its own and not with additional newlines or whitespace to indicate it is a title for the story.
   - Overall, it seems combining a few short sentences together, with a limit around 130 tokens, ensures effective translation accuracy rates.

3. Arabic, out of the languages tested, performed more poorly than other languages. This may be due to the submission text being a mismatch (could be a different variant of Arabic) so more testing is warranted.
   - In the meantime it was observed that Arabic translations improved greatly with smaller chunks of text. Thus we added a separate translation soft limit for difficult languages (Arabic).
4. As a precaution, we also tested Hebrew, which did not seem to display the same number of issues as Arabic.
   - We also examined the text inputs for Hebrew and Arabic for potential reversed character directions that may confuse the translator. Instead what we found was they were instead simply rendered differently
   in most text displays (and no special directional characters were present).