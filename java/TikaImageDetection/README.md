# Overview

This directory contains source code for the OpenMPF Tika image detection component.

Extracts images embedded in document formats (.pdf, .ppt, .doc)

For PDF documents, images will be extracted and processed per
page/slide. Detected images will be reported in the detection property
(IMAGE_FILES) in the order they were extracted. The first track (with
detection property PAGE = 0) corresponds to first page of each document by
default.

The extractor only extracts unique images once per page.
Images that are reused or duplicated in the pdf are ignored to save processing
time and avoid infinite recursion. However, modifications were made to the pdf
parser to allow for tracking of images repeatedly used across multiple pages.
If an image occurs across two or more pages it will be extracted once from the
first page then reported in the detection tracks of the following pages whenever
it found again.

Images extracted are automatically stored in a directory named after the job in
the $MPF_HOME artifacts folder. Users can toggle the job parameter
(ORGANIZE_BY_PAGE) to store each set of images in sub-directories labeled by
page number, otherwise all unique images are stored in the main job directory.
