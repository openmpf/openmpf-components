# Overview

This directory contains source code for the OpenMPF Tika image detection component.

Extracts images embedded in document formats (`.pdf`, `.ppt`, `.doc`, `.odp`)

For PDF documents, images will be extracted and processed per
page/slide. Detected images will be reported in the detection property
(`IMAGE_FILES`) in the order they were extracted. The first track (with
detection property `PAGE = 1`) corresponds to first page of each document by
default.

The extractor only extracts unique images once per page.
Images that are reused or duplicated in the PDF are ignored to save processing
time and avoid infinite recursion. However, modifications were made to the PDF
parser to allow for tracking of images repeatedly used across multiple pages.
If an image occurs across two or more pages it will be extracted once from the
first page then reported in the detection tracks of the following pages whenever
it found again. For empty pages or pages with no extracted images,
users can allow empty tracks to be reported by setting `ALLOW_EMPTY_PAGES` to true.

By default, extracted images are stored in `$MPF_HOME/share/artifacts/<job#>/tika-extracted`.
Users can set `ORGANIZE_BY_PAGE` to true to store each image in a sub-directory labeled by
page number - for example, `page-1` - and images that appear on more than one page will be
placed in a `common` directory instead.
