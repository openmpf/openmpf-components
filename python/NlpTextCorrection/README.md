# Overview
 
This repository contains source code and model data for the Natural Language Processing (NLP) text correction component. This component utilizes the symspellpy library, which is a Python port of SymSpell, to perform post-processing correction of OCR text. SymSpell uses Damerau-Levenshtein distance and Baysian probability to correct misspelled text.
Symspellpy can be found at https://github.com/mammothb/symspellpy and SymSpell can be found at https://github.com/wolfgarbe/SymSpell.
`
In order to use this component, after installing NLTK run ```python -c "import nltk; nltk.download('punkt')"```

# Custom Dictionary

Using a custom dictionary does not replace the default dictionary. The custom dictionary is merged with the default dictionary. If a word that is already in the default dictionary is included in the custom dictionary, the frequency of that word is the frequency in the default dictionary plus the frequency in the custom dictionary.

In order to use a custom dictionary, the full path to a .txt must be included as a job property under the name `CUSTOM_DICTIONARY`. The file should be formatted as a list of word freqeuncy pairs with each word on a separate line. A space should separate each word from its frequency. Here is an example of how the file should be formatted:

```the 352
of 654
and 234
to 65
a 2234
in 14
for 345
is 1
on 387
```

