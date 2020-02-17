//
// Created by mpf on 1/22/20.
//

#include <iostream>

#include "KeywordTaggingComponent.h"

using namespace MPF::COMPONENT;

int main(int argc, char *argv[]) {
    std::string job_name("tagger_test");

    Properties algorithm_properties;
    Properties media_properties;

    algorithm_properties["TAGGING_FILE"] = "text-tags.json";

    KeywordTagger tagger;

    tagger.SetRunDirectory("./plugin");
    tagger.Init();



    MPFGenericJob job(job_name, "test.txt", algorithm_properties, media_properties);

    std::vector<MPFGenericTrack> tracks;

    tagger.GetDetections(job, tracks);
}