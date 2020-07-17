#!/bin/bash

# Copy over library and model files.
cd /opt/tensorrtserver/ && ./nvidia_entrypoint.sh
cp -a /Testing_MPF_FILES/*.so* /usr/local/lib
ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so
ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so.0
ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so
ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so.0
cp /Testing_MPF_FILES/*.so /opt/tensorrtserver/
cp /Testing_MPF_FILES/plugin/TrtisDetection/lib/librequest.so /usr/local/lib/
cp -r /Testing_MPF_FILES/plugin/TrtisDetection/models /
rm /Testing_MPF_FILES/plugin/TrtisDetection/models/ip_irv2_coco/1/model.graphdef
ldconfig

# Start trtserver as a background process.
trtserver --model-store=/models 2> /Testing_MPF_FILES/server_logs.txt &

# Running test code alongside active server process.
echo "Starting server and loading model files."
cd /Testing_MPF_FILES/test/ \

# Wait for server to finish loading coco model files.
 for i in {1..10}
    do
    [[ -f "/Testing_MPF_FILES/server_logs.txt" ]] \
    && grep -q "Successfully loaded servable version" "/Testing_MPF_FILES/server_logs.txt" \
    && echo "Server ready for testing. Beginning detection tests." \
    && break

    sleep 5

    if [ $i == 10 ]
    then
        echo "Time limit reached, model load unsuccessful. Exiting tests with failure."
        exit 1
    fi
    done

# Run TRTIS tests.
./TrtisDetectionTest

