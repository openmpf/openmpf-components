# Overview

This repository contains source code and model data for the OpenMPF Caffe classification component.

# Introduction

The Caffe classification component uses the OpenCV Deep Neural Network (DNN) module to import and run trained neural network models for classification of objects in images and videos. The component supports models developed using the Caffe deep learning framework created by the Berkeley AI Research Lab [(BAIR)](http://caffe.berkeleyvision.org), formerly the Berkeley Vision and Learning Center (BVLC).

# Trained Caffe Models

The Caffe component comes with two trained Caffe models that are available when the component is registered with the workflow manager:

- The BAIR GoogLeNet model, bvlc_googlenet, from the BAIR model zoo [(http://caffe.berkeleyvision.org/model_zoo.html)](http://caffe.berkeleyvision.org/model_zoo.html), which was trained on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset.

- The Yahoo Not Suitable/Safe for Work (yahoo_nsfw) model from [https://github.com/yahoo/open_nsfw](https://github.com/yahoo/open_nsfw).

This component also supports a third model, which can be optionally downloaded:

- Reza Fuad Rachmadi's vehicle color recognition model from [https://github.com/rezafuad/vehicle-color-recognition](https://github.com/rezafuad/vehicle-color-recognition).

## Adding a new model

In order to add a trained model to the Caffe component, three files are required: the trained model, the neural network prototxt definition, and the synset text file. As an example, for the GoogLeNet model, the Caffe component supplies the files bvlc_googlenet.caffemodel, bvlc_googlenet.prototxt, and synset_words.txt. The Caffe component obtains information about the models that are available through entries in the models configuration file found in `plugin-files/config/models.cfg`.

To add a new model, follow these steps:

1. Add your .caffemodel, .prototxt, and synset files to the `plugins/config` directory. When the component is built, these files will be included in the component plugin package.
2. Add an entry to `plugins/config/models.cfg`, giving your model a name, and listing the names of the above files.

## Adding a default pipeline for your model

A set of default pipelines for a component is available to the user when the component is registered through the workflow manager web UI. To add a default pipeline for your model, you will need to add an action, a task, and a pipeline definition to the Caffe component descriptor file in `plugin-files/descriptor/descriptor.json`.

1. Add an action to the "actions" list in the descriptor file: The action must have a name and a description. It also must specify the algorithm "CAFFE", and a set of properties defining the action. The available properties are defined in the "algorithm" definition in the same file. The properties that must be defined include the MODEL_NAME, and any other properties for which something other than the default value is required.
2. Add a task to the "tasks" list, with the name of the task, a description, and the name of the action defined in step 1.
3. Add a pipeline to the "pipelines" list, with the name of the pipeline, a description, and the name of the task defined in step 2.

After the Caffe component plugin package is built and uploaded to the Component Registration page in the web UI, this new pipeline will appear in the list of available pipelines on the Job Creation page.

## Calculation of the spectral hash for a model layer

The Caffe component supports the calculation of the spectral hash for a given input image, or every frame of an input video. The spectral hash is computed according to the method outlined in ["Spectral Hashing" (Y. Weiss, et. al.)](http://papers.nips.cc/paper/3383-spectral-hashing.pdf).

The Caffe component requires a set of input parameters for the spectral hash computation. These are defined in a JSON-formatted input file. An example of a spectral hash parameter file is provided with the Caffe component source code in `plugin-files/config/bvlc_googlenet_spectral_hash.json`. The parameters in that file were derived from the GoogLeNet model training dataset (ILSVRC 2012). On a deployment machine, the correct full path name for this file is `${MPF_HOME}/plugins/CaffeDetection/config/bvlc_googlenet_spectral_hash.json`.

MATLAB code to calculate the spectral hash input parameters for a different training data set can be found [here](http://www.cs.huji.ac.il/~yweiss/SpectralHashing/), and Python code can be found [here](https://github.com/wanji/sh).


Calculation of the spectral hash is optional. It will be done in addition to the normal Caffe classification. To request this calculation, follow these steps:

1. Create an input JSON file similar to `plugin-files/config/bvlc_googlenet_spectral_hash.json`. Place your file in the same directory.

2. Create a pipeline to calculate the spectral hash. The pipeline can be added as a default pipeline, or as a custom pipeline through the web UI. In either case, you must define a new action and set the "SPECTRAL HASH FILE LIST" property to the full path to your spectral hash parameter input file. Then create a new task using this action, and and then create a new pipeline using that task, as outlined above.

   - If you choose to add a default pipeline, your spectral hash JSON file will be located in `${MPF_HOME}/plugins/CaffeDetection/config` when the component package is registered. Use this path in the value for the "SPECTRAL HASH FILE LIST" property when creating your new action.
