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

- For `.txt` files and Excel documents, all of the text will be reported in a single track with `PAGE_NUM = 1`
  and `SECTION_NUM = 1`.

- For Word documents, OpenDocument Text documents, and OpenDocument Presentation documents, all text will be reported
  with `PAGE_NUM = 1`. // TODO: This is `PAGE_NUM = 0` for TikaImageDetection

- OpenDocument Spreadsheet documents will generate one track per cell, as well as some additional tracks with
  date and time information, "Page /", and "???".