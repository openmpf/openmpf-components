# Overview

This repository contains source code for the MPF Darknet detection component.


## Building with CUDA Support
By default, The CUDA version of Darknet will be built if CUDA is installed at /usr/local/cuda. 
If you have CUDA installed, but don't want the CUDA version to be built, 
you can set the MPF_DISABLE_CUDA_BUILD environment variable to any value.



## Choosing whether a Darknet job runs on GPU or CPU
### Select the CUDA device
To run a job on the GPU the "CUDA_DEVICE_ID" algorithm property should be set to the 
ID of the GPU that the job should run on. When there is only one GPU installed this value should be 0.
When the "CUDA_DEVICE_ID" algorithm property is less than 0, the job will run on the CPU. 
If the "CUDA_DEVICE_ID" algorithm property is not provided, it will default to the value of the system
property named "detection.cuda.device.id".

### Choose whether or not to use CPU if there is a GPU issue 
The "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM" algorithm property is only used when "CUDA_DEVICE_ID" is 0 or greater. 
When "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM" is true, a job will run on the CPU if there is an issue trying to run the job 
on the GPU. When "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM" is false, the job will fail if there is a GPU issue. 
If the "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM" algorithm property is not provided, it will default to the value of the 
system property named "detection.use.cpu.when.gpu.problem".


