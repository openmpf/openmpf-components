# Overview

This directory contains source code for the OpenMPF Tika text detection component.

Supports most document formats (.txt, .pptx, .docx, .doc, .pdf, etc.) as input.
Extracts text contained in document and processes text for detected languages
(71 languages currently supported). For PDF and PowerPoint documents, text will
be extracted and processed per page/slide. The first page track (with detection
property PAGE_NUM = 1) corresponds to first page of each document by default.
Unlike PowerPoint (*.pptx) files, OpenOffice Presentation (*.odp) files can't be parsed by page.

Users can also enable metadata reporting.
If enabled by setting the job property STORE_METADATA = "true", document 
metadata will be labeled and stored as the first track.
Metadata track will not contain the PAGE_NUM or TEXT detection properties.
