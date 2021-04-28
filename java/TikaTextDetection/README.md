# Overview

This directory contains source code for the OpenMPF Tika text detection component.

This component supports most document formats (`.txt`, `.pptx`, `.docx`, `.doc`, `.pdf`, etc.) as input. It extracts
text contained in the document and processes the text for detected languages (71 languages currently supported). For PDF
and PowerPoint documents text will be extracted and processed per page (each PowerPoint slide is treated as a page).
Unlike PowerPoint (`*.pptx`) files, OpenOffice Presentation (`*.odp`) files cannot be parsed by page. The component can
still extract all of the text from Word documents and other files that cannot be parsed by page, but all of their tracks
will have a `PAGE_NUM = 1` property. Note that page numbers start at 1, not 0.

Every page can generate zero or more tracks, depending on the number of text sections in that page. A text section can
be a line or paragrah of text surrounded by newlines and/or page breaks, a single bullet point, a single table cell,
etc. In addition to `PAGE_NUM`, each track will also have a `SECTION_NUM` property. `SECTION_NUM` starts over at 1 on
each page.

Users can also enable metadata reporting. If enabled by setting the job property `STORE_METADATA = "true"`, document
metadata will be labeled and stored as the first track. Metadata track will not contain the `PAGE_NUM`, `SECTION_NUM`,
or `TEXT` detection properties.
