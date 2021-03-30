# Overview

This directory contains source code for the OpenMPF Tika text detection component.

Supports most document formats (.txt, .pptx, .docx, .doc, .pdf, etc.) as input.
Extracts text contained in document and processes text for detected languages
(71 languages currently supported). For PDF and PowerPoint documents, text will
be extracted and processed per page/slide. The first page track (with detection
property PAGE_NUM = 1) corresponds to first page of each document by default.

Users can also enable metadata reporting.
If enabled by setting the job property STORE_METADATA = "true", document 
metadata will be labeled and stored as the first track.
Metadata track will not contain the PAGE_NUM, TEXT, TAGS, or TRIGGER_WORDS detection properties.

All text extracted will also be tagged based on content.
The content tags can be set by user by providing a new context-tag file path
for the TAGGING_FILE job property.
