# Overview

This repository contains source code for Open Media Processing Framework (OpenMPF) components 
licensed under an Apache 2.0 or compatible license.


# Building the C++ Components
* In order to build the C++ components you must first install the 
  [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk).
* cd into the `openmpf-components/cpp` directory.
* Run the following commands:
```
mkdir build
cd build
cmake3 ..
make install
```
* The built plugin packages will be created in `openmpf-components/cpp/build/plugin-packages`.

### Building Individual C++ Components
If you would like to only build a single component, you can cd into that component's directory and run the
build commands listed above.



# Building the Java Components
* In order to build the Java components you must first install the 
  [OpenMPF Java Components SDK](https://github.com/openmpf/openmpf-java-component-sdk)
* cd into the `openmpf-components/java` directory.
* Run `mvn package`.
* The built plugin packages will be in each components' `target/plugin-packages` directory.

### Building Individual Java Components
If you would like to only build a single components, you can cd into that component's directory and run `mvn package`.
