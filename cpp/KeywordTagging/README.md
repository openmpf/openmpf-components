# Text Tagging

All text extracted from an image can also be tagged using regular expression (regex) patterns
specified in a JSON tagging file. By default this file is located in the
`config` folder as `text-tags.json`. Users can provide an alternate path to a
tagging file of their choice. English and foreign regex patterns following UTF-8
encoding are supported. Regex searches are case-insensitive.

In the tagging file, users can specify regex patterns using the [Boost library
regex operators](https://cs.brown.edu/~jwicks/boost/libs/regex/doc/syntax.html).
Of note, the `\W` non-word operator and `\b` word-break operator may prove useful.
Note that these must be escaped as `\\W` and `\\b` in JSON.

Regex tags in the JSON tagging file can be entered as follows:

```
    "TAGS_BY_REGEX": {
        "vehicle": [
            {"pattern": "auto"},
            {"pattern": "(\\b)bike(\\b)"},
            {"pattern": "(\\b)bus(\\b)", "caseSensitive": true}
        ]
        "next-tag-category" : [
            ...
        ]
    }
```

Where each `"pattern"` specifies the regex used for tagging.
To enable case-sensitive regex tag search, set the `"caseSensitive"` flag to true for each
regex pattern that requires case sensitivity. For example:

```
    {"pattern": "Financial", "caseSensitive": true}
```

Will search for words containing "Financial" with the first letter capitalized.
On the other hand the following patterns:

```
    {"pattern" :"Financial", "caseSensitive": false}
    {"pattern" :"Financial"}
```

Will search for "financial", "Financial", "FINANCIAL", and any other variation
of "financial" in terms of capitalization.

Phrases containing words separated by **zero** or more whitespace and/or
punctuation characters can be represented using `\\W*`. For example, the
`Hello(\\W*)World` regex pattern will match:

* "Hello World"
* "HelloWorld"
* "Hello.World"
* "Hello. &$#World"

Phrases containing words separated by **one** or more whitespace and/or
punctuation characters can be represented using `\\W+`.

For example, the `Hello(\\W+)World` regex pattern will match:

* "Hello World"
* "Hello.World"
* "Hello. &$#World"

But not:

* "HelloWorld"

Adding `\\b` to the start or end of a regex pattern will reject any word
characters attached to the start or end of the pattern being searched. For
example, `(\\b)search(\\W+)this(\\W+)phrase(\\b)` will match:

* "search this phrase"
* "search  this, phrase"

But not:

* "research this phrase"
* "search this phrasejunk"

Removing the leading and trailing `\\b` will allow these phrases to be matched, excluding the extraneous leading/trailing characters.

To escape and search for special regex characters use double slashes `\\` in front of each special character.
To escape and search for a single backslash in text, users have to specify `\\\\` as the regex pattern.

For example, to search for periods we use `\\.` rather than `.`, so the regex pattern becomes `(\\b)end(\\W+)of(\\W+)a(\\W+)sentence\\.`. Note that the `.` symbol is typically used in regex to match any character, which is why we use `\\.` instead.

The OCR'ed text will be stored in the `TEXT` output property. Each detected tag will be stored in `TAGS`, separated by semicolons. The substring(s) that triggered each tag will be stored in `TRIGGER_WORDS`. For each trigger word the substring index range relative to the `TEXT` output will be stored in `TRIGGER_WORDS_OFFSET`. Because the same trigger word can be encountered multiple times in the `TEXT` output, the results are organized as follows:

* `TRIGGER_WORDS`: Each distinct trigger word is separated by a semicolon followed by a space. For example: `TRIGGER_WORDS=trigger1; trigger2`
    * Because semicolons can be part of the trigger word itself, those semicolons will be encapsulated in brackets. For example, `detected trigger with a ;` in the OCR'ed `TEXT` is reported as `TRIGGER_WORDS=detected trigger with a [;]; some other trigger`.
* `TRIGGER_WORDS_OFFSET`: Each group of indexes, referring to the same trigger word reported in sequence, is separated by a semicolon followed by a space. Indexes within a single group are separated by commas.
    * Example `TRIGGER_WORDS=trigger1; trigger2`, `TRIGGER_WORDS_OFFSET=0-5, 6-10; 12-15`, means that `trigger1` occurs twice in the text at the index ranges 0-5 and 6-10, and `trigger2` occurs at index range 12-15.

Note that all `TRIGGER_WORD` results are trimmed of leading and trailing whitespace, regardless of the regex pattern used. The respective `TRIGGER_WORDS_OFFSET` indexes refer to the trimmed substrings.

