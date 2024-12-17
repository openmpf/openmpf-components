# Overview

This repository contains source code for the OpenMPF No Language Left Behind component. This component is based on [Meta's No Language Left Behind Project](https://ai.meta.com/research/no-language-left-behind/)and uses the [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) model.

This component translates the input text from a given source language to English. The source language can be provided as a job property, or be indicated in the detection properties from a feed-forward track.


# Job Properties
The below properties can be optionally provided to alter the behavior of the component.

- `FEED_FORWARD_PROP_TO_PROCESS`: Controls which properties of the feed-forward track or detection are considered for translation. This should be a comma-separated list of property names. The default properties are `"TEXT,TRANSCRIPT"`. Each named property in the list that is present is translated and the resulting translations will be added to the feed forward track. The defaul properties will be saved as `TEXT TRANSLATION` or `TRANSCRIPT TRANSLATION` depending on which (or both) is present. For other property names that are provided, the tranlastion property will follow the same convention and tranlastions will be saved to a `[PROPERTY NAME] TRANSLATION`.

- `LANGUAGE_FEED_FORWARD_PROP`: Is an optional feed-forward property that comes from an earlier stage in a pipeline, indicating which language to translate from. The default values are `"ISO_LANGUAGE, DECODED_LANGUAGE, LANGUAGE"`. The ISO 639-3 standard is the expected input for any of these fields. If the first property listed is present, then that property will be used. If it's not, then the next property in the list is considered. If none are present, the languge to translate from will be determiend by the DEFAULT_SOURCE_LANGUAGE property instead.

- `SCRIPT_FEED_FORWARD_PROP`: Is an optional feed-forward property that is set from an earleir stage in the a pipeline, indicating which source script to use during translation. The default values are `"ISO_SCRIPT, DECODED_SCRIPT, SCRIPT"`. The ISO 15925 is expected as the input for these fields. The first property that is found in the list will be used, otherwise the the DEFAULT_SOURCE_SCRIPT property will be used to select the script to be used for translation.

- `DEFAULT_SOURCE_LANGUAGE`: The default source language to use if none of the property names listed in `LANGUAGE_FEED_FORWARD_PROP` are present in a feed-forward track or detection. Values should be in the ISO 639-3 standard.

- `DEFAULT_SOURCE_SCRIPT`: The default source script to use if none of the property names listed in the `SCRIPT_FEED_FORWARD_PROP` are present in a feed-forward track or detection. Values should be in the ISO 15925 standard.

- `TARGET_LANGUAGE`: Optional property to define a language to translate to. This value defaults to english if the property is not present.

- `TARGET_SCRIPT`: Optional property to define a script to be used in tranlating a language to. This value defaults to latin.


# Language Identifiers
The following are the ISO 639-3 and ISO 15925 codes, and their corresponding languages which Nllb can translate.

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
