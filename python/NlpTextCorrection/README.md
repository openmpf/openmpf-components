# Overview
 
This repository contains source code and model data for the Natural Language Processing (NLP) text correction component. This component utilizes the cython_hunspell library, which is a Python port of Hunspell, to perform post-processing correction of OCR text. Hunspell is the spell checker of LibreOffice, OpenOffice.org, Mozilla Firefox 3 & Thunderbird, and Google Chrome.
Hunspell can be found at https://github.com/hunspell/hunspell and cython_hunspell can be found at 
https://github.com/MSeal/cython_hunspell


# Custom Dictionary

Using a custom dictionary does not replace the default dictionary. The custom dictionary is merged with the default dictionary. 

In order to use a custom dictionary, the full path to a .dic must be included as a job property under the name `CUSTOM_DICTIONARY`. The file should be formatted as a list of words, with one word per line. The first line contains the approximate word count. Here is an example of how the file should be formatted:

```
3
hello
try
work
```

You may also use an affix file (.aff) to specify flags used during spell checking. 

For more information on dictionary files and affix files for hunspell, see https://linux.die.net/man/4/hunspell.

# Full suggestions output
The default behavior of this component is for the corrected text to use the first suggested correction provided by Hunspell. If you want the output to contain the full suggestions list set `FULL_TEXT_CORRECTION_OUTPUT` to true.

