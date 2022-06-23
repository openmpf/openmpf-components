# Overview

This directory contains source code for the OpenMPF Tika image detection component.

Extracts images embedded in document formats (`.pdf`, `.ppt`, `.doc`)

In general, the Tika parsers will extract unique images, once per track.
Each track will contain a `PAGE_NUM` property specifying where the
image is embedded. The first page corresponds to a `PAGE_NUM` value of `1`
by default. Multiple pages can be reported in a PDF document and are
separated by semicolons.

For instance, `PAGE_NUM = 1; 2; 4`, would indicate the embedded image
appears on pages 1, 2, and 4 of a PDF document. The path where the
extracted image is stored will be reported in the `DERIVATIVE_MEDIA_TEMP_PATH`
property.

Images extracted from other formats (i.e. `.ppt`) tend to be embedded either
before or after the text content and would appear with `PAGE_NUM`
indicating either the first or last page of that document.

By default, extracted images are stored in `$MPF_HOME/share/tmp/derivative-media/<job#>/<uuid>/tika-extracted`.
Users can set `ORGANIZE_BY_PAGE` to true to store each image in a sub-directory labeled by
page number - for example, `page-1` - and images that appear on more than one page will be
placed in a `common` directory instead.
