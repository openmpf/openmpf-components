# Overview

This repository contains source code for the OpenMPF CLIP detection component. CLIP (Contrastive Language-Image Pre-Training) was developed by OpenAI and published in 2021. https://arxiv.org/abs/2103.00020

When building on a host without a GPU, the following warning is expected:
```
UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)
  return torch._C._cuda_getDeviceCount() > 0
```

# Job Properties

The following are the properties that can be specified for the component. Each property has a default value and so none of them necessarily need to be specified for processing jobs.

- `NUMBER_OF_CLASSIFICATIONS`: Specifies how many of the top classifications you want to return. The default value is set to 1, and so you'll only see the classification with the greatest confidence.

- `CLASSIFICATION_PATH`: If specified, this allows the user to give the component a file path to their own list of classifications in a CSV file, if the COCO or ImageNet class lists aren't of interest. See below for the formatting that's required for that file.

- `CLASSIFICATION_LIST`: Specifies whether the user wants to use the COCO or ImageNet classification list, by specifying 'coco' or 'imagenet', respectively. By default, this is set to 'coco'. Also this property is overridden if a `CLASSIFICATION_PATH` is given.

- `TEMPLATE_PATH`: If specified, this allows the user to give the component a file path to their own list of templates. See below for the formatting that's required for that file. The OpenAI developers admitted that the process of developing templates was a lot of trial and error, so feel free to come up with your own!

- `NUMBER_OF_TEMPLATES`: There are three template files that are included in the component, with the number of templates in each being 1, 7, and 80. The one template is a basic template, while the 7 and 80 come from the OpenAI team when trying to [improve performance](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb) on the ImageNet dataset. The default value is 80, while 1 and 7 are the only other valid inputs. Also this property is overridden if a `TEMPLATE_PATH` is specified.

- `ENABLE_CROPPING`: A boolean toggle to specify if the image is to be cropped into 144 images of size 224x224 which cover all areas of the original. By default, this is set to true. This technique is described Section 7 of the paper "[Going deeper with convolutions](https://arxiv.org/abs/1409.4842)" from Szegedy, et al. 

- `ENABLE_TRITON`: A boolean toggle to specify whether the component should use a Triton inference server to process the image job. By default this is set to false.

- `TRITON_SERVER`: Specifies the Triton server `<host>:<port>` to use for inferencing. By default, this is set to 'clip-detection-server:8001'.

## Detection Properties

Returned `ImageLocation` objects have the following members in their `detection_properties`:

| Property Key                     | Description 
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------
| `CLASSIFICATION`                 | The classification returned from the CLIP model with the highest similarity.
| `CLASSIFICATION CONFIDENCE LIST` | A list of the highest confidences returned by the model. The number of confidences is specified by `NUMBER_OF_CLASSIFICATIONS`.
| `CLASSIFICATION LIST`            | A list of the classes with highest confidence returned by the model. The number of classes is specified by `NUMBER_OF_CLASSIFICATIONS`.

# Custom Templates

When tuning the CLIP model, it is important to have appropriate templates for what you're trying to classify. In order to write the file, put one template on each line. Use a pair of brackets, {}, where the potential classifications need to be placed. See below for example templates.
```
A photograph of a {}.
A {} in an open field.
```

# Custom Classifications

The need for custom classifications arose when training on the ImageNet classifications, where any different class can have many equivalent names. For example, one of the classes is "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias". We found the model to be most performant when given a single representative class title. For this case, 'great white shark' makes the most sense. The `imagenet_classification_list.csv` file gives representative titles for each class, adapted from .ipynb files on the CLIP GitHub page.

As for the format of the CSV file, it has two columns. The first being the representative name, and the second being the full name of the class. The representative name is what goes inside the brackets, {}, of the templates, and the full name is what will be used when displaying results. Below are a couple of examples of rows from the ImageNet classifications. Note that in the first example, quotes are put around the full classification name so they're easier to read and so that those commas aren't confused for the separator.

```
tench,"tench, Tinca tinca"
kite (bird of prey),kite
magpie,magpie
```