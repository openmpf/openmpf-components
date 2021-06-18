# OpenMPF Components

Welcome to the Open Media Processing Framework (OpenMPF) Component Project!

## What is the OpenMPF?

OpenMPF provides a platform to perform content detection and extraction on bulk multimedia, enabling users to analyze, search, and share information through the extraction of objects, keywords, thumbnails, and other contextual data.

OpenMPF enables users to build configurable media processing pipelines, enabling the rapid development and deployment of analytic algorithms and large-scale media processing applications.

### Search and Share

Simplify large-scale media processing and enable the extraction of meaningful content

### Open API

Apply cutting-edge algorithms such as face detection and object classification

### Flexible Architecture

Integrate into your existing environment or use OpenMPF as a standalone application

## Overview

This repository contains source code for Open Media Processing Framework (OpenMPF) components licensed under an Apache 2.0 or compatible license.

## Where Am I?

- [Parent OpenMPF Project](https://github.com/openmpf/openmpf-projects)
- [OpenMPF Core](https://github.com/openmpf/openmpf)
- Components
    * [OpenMPF Standard Components](https://github.com/openmpf/openmpf-components) ( **You are here** )
    * [OpenMPF Contributed Components](https://github.com/openmpf/openmpf-contrib-components)
- Component APIs:
    * [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk)
    * [OpenMPF Java Component SDK](https://github.com/openmpf/openmpf-java-component-sdk)
    * [OpenMPF Python Component SDK](https://github.com/openmpf/openmpf-python-component-sdk)
- [OpenMPF Build Tools](https://github.com/openmpf/openmpf-build-tools)
- [OpenMPF Web Site Source](https://github.com/openmpf/openmpf.github.io)
- [OpenMPF Docker](https://github.com/openmpf/openmpf-docker)

## Getting Started

### Creating New Components

- [How to Create a C++ Component](https://openmpf.github.io/docs/site/CPP-Batch-Component-API/index.html#getting-started)
- [How to Create a Python Component](https://openmpf.github.io/docs/site/Python-Batch-Component-API/index.html#how-to-create-a-python-component)
- [How to Create a Java Component](https://openmpf.github.io/docs/site/Java-Batch-Component-API/index.html#getting-started)

### Building a Component Using Docker

The following steps apply to components implemented in any language:

- `cd` into the component directory that contains the `Dockerfile`.
- Run the following command where:
    - `<openmpf_component_name>` is the name of the component image you want to generate. For example, "openmpf_ocv_face_detection". For a list of component image names, refer to [docker-compose.components.yml](https://github.com/openmpf/openmpf-docker/blob/master/docker-compose.components.yml).
    - `<tag-name>` is the tag applied to the generated image. If you're unsure, set this to "latest".
    - `<version>` is the version of the component base image that will be pulled from [Docker Hub](https://hub.docker.com/u/openmpf). If you're unsure, set this to "latest".
```
DOCKER_BUILDKIT=1 docker build . -t <openmpf_component_name>:<tag-name> --build-arg BUILD_REGISTRY=openmpf/ --build-arg BUILD_TAG=<version>
```

### Building C++ Components Outside of Docker

- In order to build the C++ components you must first install the [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk#build-and-install-the-component-sdk).
- `cd` into the `openmpf-components/cpp` directory.
- Run the following commands:
```
mkdir build
cd build
cmake3 ..
make install
```

- The built plugin packages will be created in: `openmpf-components/cpp/build/plugin-packages`.

**Building an Individual C++ Component**

If you would like to only build a single component, you can `cd` into that component's directory and run the build commands listed above.

### Building a Python Component Outside of Docker

- In order to build the Python components you must first install the [OpenMPF Python Component SDK](https://github.com/openmpf/openmpf-python-component-sdk#build-and-install-the-component-sdk).
- `cd` into the Python component directory that contains the `setup.py` file
- Run the following commands to install the component in a virtual environment:
```
python3 -m venv venv
. ./venv/bin/activate
pip3 install --upgrade pip
pip3 install openmpf-python-component-sdk/detection/api
pip3 install openmpf-projects/openmpf-python-component-sdk/detection/component_util
pip3 install .
```

### Building Java Components Outside of Docker

- In order to build the Java components you must first install the [OpenMPF Java Components SDK](https://github.com/openmpf/openmpf-java-component-sdk#build-and-install-the-component-sdk).
- `cd` into the `openmpf-components/java` directory.
- Run `mvn package`.
- The built plugin packages will be in each components' `target/plugin-packages` directory.

**Building an Individual Java Component**

If you would like to only build a single component, you can `cd` into that component's directory and run `mvn package`.

### Project Website

For more information about OpenMPF, including documentation, guides, and other material, visit our website: [https://openmpf.github.io/](https://openmpf.github.io/)

### Project Workboard

For a latest snapshot of what tasks are being worked on, what's available to pick up, and where the project stands as a whole, check out our [workboard](https://github.com/orgs/openmpf/projects/3).
