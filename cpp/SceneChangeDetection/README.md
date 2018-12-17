# Overview

This repository contains source code for the OpenMPF video scene change detection
component.

Detects and segments a given video by scenes. Each scene change is detected using
histogram comparison, edge comparison, brightness (fades outs), and overall
hue/saturation/value differences between adjacent frames. Users can toggle each
type of scene change detection technique using (DO_HIST, DO_THRS, DO_EDGE, DO_CONT)
respectively as well as threshold parameters for each detection method.

For a video with a given frame interval  (ex.[0,200]), this module will report
each detected scene as a separate track with start/end frames of each track
representing video frames associated with that scene (ex.scene 1: [0,41]
scene 2: [42,200]).

Users may also toggle USE_MIDDLE_FRAME to either store a single MPFImageLocation
representing each scene track or all frames associated with each track. When set
to true (default), only the middle frame of each scene will be stored
(ex. frame 20 for a scene ranging from 0 to 40).
