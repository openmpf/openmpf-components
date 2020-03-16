Trtisdetection status:

* Docker containers for client and server are now working.
* Need to test out TrtisDetection component, sample binary.
* Component is close to 1 GB in size (model files are large, might not want to include that in the official repo).
* Double check code contains no sensitive words or phrases before uploading to public repo.


Most of the current work has been testing and building the dockerfile in docker\_for\_dev.

Progress and notes so far:

1. The dockerfile in docker\_for\_dev builds off of the baseline OpenMPF image (openmpf_build).
   * As a result, I started with the OpenMPF Build Image first, and followed instructions for building the workflow manager as well.
   * For some reason, I kept running into python-component-api test issues when testing the MPF docker workflow manager. They
   were also encountered in my OpenMPF builds in my development environment.
   * There were likely some dependencies that were missing or a possible version mismatch (errors stated a python dependency was missing, or some function calls were invalid).
   * Ffmpeg was one of the python libraries that seemed to have a version mismatch or related issue.

   This issue was resolved by skipping the related python tests and following the remaining build steps.
   It does not seem related to the trtisdetection branch so I moved on to the next step once the images were built.

2. Next, I proceeded to setup the client and trtserver containers. I started with the dockerfile in docker\_for\_dev, which builds the openmpf-dev image and container.
   I ran into more build issues and proceeded to update the dockerfile.

3. Updates for openmpf-dev dockerfile:
   * The original version of this dockerfile I received downloads and installs the latest available git source code releases for each of the dependencies (TensorRT-Inference-Server, grpc). They were not set to specific branches or version numbers.
   * I've modified the instructions to build from a specific release for each of the dependencies to avoid future dependency conflicts. They are still the latest available releases for the moment.
   * Also added in steps for manual build and installation of curl (version: 7_67_0) as cmake failed to properly locate and setup the yum installed curl-devel packages.
     Previous attempts to point cmake3 to the libcurl lib/bin locations still failed, even though find_package() for curl worked. Might have been an incompatible version of curl.
   * Resolved additional version complaints from cmake3 for the TensorRT client cmakelist. Swapped line "project (trtis-clients)" with "project(trtis-clients VERSION "0.0.0")".
   * Resolved OpenCV3.3.0 vs OpenCV3.4.7 dependency conflict by swapping to "develop" branch for the openmpf-cpp-component-sdk repo.

   Andy also left some fixes for locating protobuf ("#Fix lower case protobuf package name.") and fixing src header references. I've kept those fixes in this dockerfile as well.
   After these changes were made the openmpf-dev image was successfully built.

4. Running docker-compose on client and trtserver containers:
   * I didn't run into significant issues here.
   * After I updated the docker-compose.yml files (changed volume locations in client), I was able to launch both client and server containers without any problems.
   * Note: I did have to comment out "#runtime: nvidia" for both docker-compose.yml files as they were not recognized (likely due to a different version of docker-compose).


Remaining steps (todo):

   * Update local environment dependencies to build test scripts and trtisdetection component. (Likely will reuse some of the steps from the client docker container).
   * Run test scripts & other models on available data.
   * Double check code for sensitive information.
   * Remove large files (there appears to be some model files included, such as ip\_irv2\_coco). Not sure of all of these should be uploaded.
   * Submit component PR to public OpenMPF git repository.




