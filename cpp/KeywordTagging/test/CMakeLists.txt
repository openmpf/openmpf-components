cmake_minimum_required(VERSION 3.6)
project(keyword-tagger-tests)

set(CMAKE_CXX_STANDARD 11)

find_package(GTest)
if (${GTEST_FOUND})
    enable_testing()
    find_package(GTest REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)
    add_executable(KeywordTaggingTest test_keyword_tagging.cpp)
    target_link_libraries(KeywordTaggingTest mpfKeywordTagging mpfComponentTestUtils GTest::GTest GTest::Main)

    add_test(NAME KeywordTaggingTest COMMAND KeywordTaggingTest)

    # Install test images and videos.
    file(COPY data/ DESTINATION data)
    file(COPY config/ DESTINATION config)
endif()