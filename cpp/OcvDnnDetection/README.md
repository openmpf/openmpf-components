# Overview

This repository contains source code and model data for the OpenMPF OpenCV Deep Neural Network (DNN) classification component.

# Introduction

The OpenCV DNN classification component uses the OpenCV DNN module to import and run trained neural network models for classification of objects in images and videos. The component supports models developed using the [Caffe](http://caffe.berkeleyvision.org), and [TensorFlow](https://www.tensorflow.org/) deep learning frameworks. The OpenCV DNN module additionally supports [Torch](http://torch.ch/), [Darknet](https://pjreddie.com/darknet/), [DLDT](https://software.intel.com/openvino-toolkit), and [ONNX](https://onnx.ai/). These frameworks have not been tested with OpenMPF's OpenCV DNN classification component.

# Trained Models

The OpenCV DNN classification component comes with two trained Caffe models and one TensorFlow model, all of which are available when the component is registered with the Workflow Manager:

* The BAIR GoogLeNet model, bvlc_googlenet, from the BAIR model zoo [(http://caffe.berkeleyvision.org/model_zoo.html)](http://caffe.berkeleyvision.org/model_zoo.html), which was trained on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset.
* The Yahoo Not Suitable/Safe for Work (yahoo_nsfw) model from [https://github.com/yahoo/open_nsfw](https://github.com/yahoo/open_nsfw).
* A model trained in-house for vehicle color estimation, vehicle_color.

This component also supports a third model, which can be optionally downloaded:

* Reza Fuad Rachmadi's vehicle color recognition model from [https://github.com/rezafuad/vehicle-color-recognition](https://github.com/rezafuad/vehicle-color-recognition).

# Adding a new model

In order to add a trained model to the OpenCV DNN classification component, the following three files should be supplied:

* The model binary file, containing the neural network weights. The following extensions are expected for models from different frameworks:
  - `.caffemodel` (Caffe)
  - `.pb` (TensorFlow)
  - `.t7` or `.net` (Torch)
  - `.weights` (Darknet)
  - `.bin` (DLDT)
  - `.onnx` (ONNX)

* The synset text file, which lists the class labels associated with the model output layer indices. This is expected to be a .txt file.

* The neural network config file, a text file containing the network configuration. This file is not required for all frameworks, as the model binary may define the network configuration. The following extensions are expected for models from different frameworks:
  - `.prototxt` (Caffe)
  - `.pbtxt` (TensorFlow, optional)
  - `.cfg` (Darknet)
  - `.xml` (DLDT)

The OpenCV DNN classification component obtains information about the models that are available through entries in the models configuration file found in `plugin-files/models/models.ini`.

To add a new model, follow these steps:

1. Add your model binary, synset file, and config file (if necessary) to the `plugins/models` directory. When the component is built, these files will be included in the component plugin package.
2. Add an entry to `plugins/models/models.ini`, giving your model a name, and listing the names of the above files.

## Adding a default pipeline for your model

A set of default pipelines for a component is available to the user when the component is registered through the Workflow Manager web UI. To add a default pipeline for your model, you will need to add an action, a task, and a pipeline definition to the OpenCV DNN classification component descriptor file in `plugin-files/descriptor/descriptor.json`.

1. Add an action to the "actions" list in the descriptor file: The action must have a name and a description. It also must specify the algorithm "DNNCV", and a set of properties defining the action. The available properties are defined in the "algorithm" definition in the same file. The properties that must be defined include the `MODEL_NAME`, and any other properties for which something other than the default value is required.
2. Add a task to the "tasks" list, with the name of the task, a description, and the name of the action defined in step 1.
3. Add a pipeline to the "pipelines" list, with the name of the pipeline, a description, and the name of the task defined in step 2.

After the OpenCV DNN classification component plugin package is built and uploaded to the Component Registration page in the web UI, this new pipeline will appear in the list of available pipelines on the Job Creation page.

## Adding a model at runtime

The above sections describe how to add a new model to the OpenCV DNN classification component before the component is packaged. If you have an instance of the Workflow Manager already running, and the component is already registered, here are the steps for using a new model post-deployment:

1. Create the `[MODELS_DIR_PATH]/OcvDnnDetection` directory where`MODELS_DIR_PATH` is defined by property with the same name in the `descriptor.json` file. Unless set to something else, by default the value is determined by the `detection.models.dir.path` system property, which is set to `$MPF_HOME/share/models` by default. This makes the full path `$MPF_HOME/share/models/OcvDnnDetection`.

2. Place your updated `models.ini` file, and the files associated with your new model, in the above directory.

3. Create a pipeline for your model using the Pipeline Creation web UI. Make sure to specify the new `MODEL_NAME`.

In general, the OpenCV DNN classification component will check the directory in step 1 for `models.ini` and model files listed therein. The component will prioritize using those files over the ones that were installed by registering the component plugin package. This allows you to add new models and/or use updated versions of models at runtime by overriding what's provided by default.


## Calculation of the spectral hash for a model layer

The OpenCV DNN classification component supports the calculation of the spectral hash for a given input image, or every frame of an input video. The spectral hash is computed according to the method outlined in ["Spectral Hashing" (Y. Weiss, et. al.)](http://papers.nips.cc/paper/3383-spectral-hashing.pdf).

The OpenCV DNN classification component requires a set of input parameters for the spectral hash computation. These are defined in a JSON-formatted input file. An example of a spectral hash parameter file is provided with the OpenCV DNN classification component source code in `plugin-files/models/bvlc_googlenet_spectral_hash.json`. The parameters in that file were derived from the GoogLeNet model training dataset (ILSVRC 2012). On a deployment machine, the correct full path name for this file is `${MPF_HOME}/plugins/OcvDnnDetection/models/bvlc_googlenet_spectral_hash.json`.

MATLAB code to calculate the spectral hash input parameters for a different training data set can be found [here](http://www.cs.huji.ac.il/~yweiss/SpectralHashing/), and Python code can be found [here](https://github.com/wanji/sh).


Calculation of the spectral hash is optional. It will be done in addition to the normal OpenCV DNN classification. To request this calculation, follow these steps:

1. Create an input JSON file similar to `plugin-files/models/bvlc_googlenet_spectral_hash.json`. Place your file in the same directory.

2. Create a pipeline to calculate the spectral hash. The pipeline can be added as a default pipeline, or as a custom pipeline through the web UI. In either case, you must define a new action and set the `SPECTRAL_HASH_FILE_LIST` property to the full path to your spectral hash parameter input file. Then create a new task using this action, and then create a new pipeline using that task, as outlined above.

   - If you choose to add a default pipeline, your spectral hash JSON file will be located in `${MPF_HOME}/plugins/OcvDnnDetection/models` when the component package is registered. Use this path in the value for the "SPECTRAL HASH FILE LIST" property when creating your new action.
