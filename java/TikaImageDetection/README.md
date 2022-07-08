# Overview

This directory contains source code for the OpenMPF Tika image detection component.

This component extracts images embedded in document formats, such as PDF (`.pdf`), PowerPoint (`.pptx`),
Word (`.docx`), OpenDocument Presentation (`.odp`), and OpenDocument Text (`.odt`) documents.

In general, the Tika parsers will extract unique images, once per track. Each track will contain a `PAGE_NUM` property
specifying where the image is embedded. Note that page numbers start at 1, not 0, unless otherwise noted.

The path where the extracted image is stored will be reported in the `DERIVATIVE_MEDIA_TEMP_PATH` property for each
track.

By default, extracted images are stored in `$MPF_HOME/share/tmp/derivative-media/<job#>/<uuid>/tika-extracted`. Users
can set `ORGANIZE_BY_PAGE` to true to store each image in a sub-directory labeled by page number (for example, `page-1`)
. Images that appear on more than one page will be placed in a `common` directory instead.

Note that the OpenMPF Workflow Manager will move those files to a more persistent storage location reported in
the JSON output object for each piece of derivative media with the job is complete.

# Format-Specific Behaviors

The following format-specific behaviors were observed using Tika 1.28.1 on Ubuntu 20.04:

- For PDF files, the first page corresponds to a `PAGE_NUM` value of `
  1`. Multiple pages can be reported in a PDF document and are separated by semicolons. For
  example, `PAGE_NUM = 1; 2; 4`
  , would indicate the embedded image appears on pages 1, 2, and 4 of a PDF document.

- For PowerPoint documents, all images will be reported on the last page of the document. For example, if the document
  is 5 pages in length, all images will be reported with `PAGE_NUM = 5`.

- For Word documents, OpenDocument Text documents, and OpenDocument Presentation documents, all images will be reported
  with `PAGE_NUM = 0`.

- OpenDocument Presentation documents will generate a thumbnail / preview `.png` of the content of the last modified
  slide, including text, even if it's blank.

- OpenDocument Text documents will generate a thumbnail / preview `.png` of the content of the first page, including
  text, even if it's blank.
