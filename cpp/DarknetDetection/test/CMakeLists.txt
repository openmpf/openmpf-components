cmake_minimum_required(VERSION 3.6)
project(darknet-detection-tests)

set(CMAKE_CXX_STANDARD 11)

find_package(GTest)
if (${GTEST_FOUND})

    enable_testing()
    find_package(GTest REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)
    add_executable(DarknetDetectionTest test_darknet_detection.cpp)
    target_link_libraries(DarknetDetectionTest mpfDarknetDetection mpfComponentTestUtils GTest::GTest GTest::Main)

    add_test(NAME DarknetDetectionTest COMMAND DarknetDetectionTest)

    # Install test images and videos
    file(COPY data/ DESTINATION data)
endif()