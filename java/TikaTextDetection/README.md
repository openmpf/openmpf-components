# Overview

This directory contains source code for the OpenMPF Tika text detection component.

Supports most document formats (.txt, .pptx, .docx, .doc, .pdf, etc.) as input.
Extracts text contained in document and processes text for detected languages (71 languages currently supported).
For PDF and PowerPoint documents, text will be extracted and processed per page/slide.
All text extracted will also be tagged based on content.
The content tags can be set by user (provide a new context-tag file path for the TAGGING_FILE job property).
