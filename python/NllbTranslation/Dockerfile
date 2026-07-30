# syntax=docker/dockerfile:1.4

#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

ARG BUILD_REGISTRY
ARG BUILD_TAG=latest

# To use only CPU resources, update
# below line to BUILD_TYPE=cpu
ARG BUILD_TYPE=gpu

# Convert the upstream checkpoint to a CTranslate2 model at build time, at a
# precision keyed to BUILD_TYPE: gpu -> float16, cpu -> int8. There is no
# published CTranslate2 build of NLLB-200-3.3B at fp16, so conversion is
# unavoidable for the GPU target; doing both from one source keeps a single
# provenance and one pinned revision behind every image.
#
# Use a base image we did not build so that we do not re-download and re-convert
# the model every time openmpf_python_component_build changes. It cannot be
# python:3.12-alpine (which the HF-checkpoint build used): the converter reads
# the checkpoint's pytorch_model-*.bin shards through torch, and torch publishes
# no musl wheels. The CPU-only torch wheel is sufficient -- conversion never
# touches a GPU -- and avoids pulling the multi-GB CUDA build.
FROM python:3.12 AS convert_model

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir 'huggingface_hub[cli]' ctranslate2 transformers sentencepiece

ARG BUILD_TYPE
ARG MODEL_NAME=facebook/nllb-200-3.3B
ARG MODEL_REVISION=1a07f7d195896b2114afcb79b7b57ab512e7b43e

# --copy_files is load-bearing, not a nicety: a bare conversion emits only
# config.json, model.bin and shared_vocabulary.json. sentencepiece.bpe.model is
# the tokenizer the component loads (verified byte-identical to the FLORES SPM
# this component used to download separately, md5
# 05c551ae7955b3980d5a9d044eb09d70), and the tokenizer.json trio keeps
# transformers' AutoTokenizer usable as an alternative backend.
#
# Quantization: bare `int8` resolves to int8_float32. On CPU that is the only
# int8 mode CTranslate2 offers and is exactly what we want. On GPU it would be a
# trap -- accurate but with no tensor-core speedup -- which is why the GPU path
# converts to float16 instead. Do not "fix" the CPU path to int8_float16:
# CTranslate2 raises ValueError for that compute type on CPU.
RUN if [ "$BUILD_TYPE" = "cpu" ]; then QUANT=int8; else QUANT=float16; fi \
    && echo "Converting $MODEL_NAME -> CTranslate2 (quantization=$QUANT)" \
    && ct2-transformers-converter \
        --model "$MODEL_NAME" --revision "$MODEL_REVISION" \
        --output_dir /models/nllb-200-3.3B-ct2 \
        --quantization "$QUANT" --low_cpu_mem_usage \
        --copy_files sentencepiece.bpe.model tokenizer.json \
                     tokenizer_config.json special_tokens_map.json


FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} AS build

ARG BUILD_TYPE
RUN <<eot
    cd /home/mpf/openmpf-projects/openmpf-python-component-sdk/detection/nlp_text_splitter
    if [[ "${BUILD_TYPE}" == "gpu" ]]; then
        ./install.sh --gpu
    else
        ./install.sh
    fi
eot

RUN pip install --no-cache-dir 'accelerate>=0.26.0' 'numpy<2'

ARG RUN_TESTS=false
RUN --mount=target=.,readwrite \
    --mount=from=convert_model,src=/models,target=/models \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_nllb_translation.py; fi



FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG} AS cpu_component

# What the build intended, so the component can flag a mismatch at load time
# (e.g. a float16 model silently up-converted to float32 on a CPU-only host).
ENV NLLB_EXPECTED_COMPUTE_TYPE=int8_float32

COPY --from=convert_model /models /models
COPY --from=build /opt/wtp /opt/wtp

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV
COPY --from=build $PLUGINS_DIR/NllbTranslation $PLUGINS_DIR/NllbTranslation

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF No Language Left Behind Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"

FROM cpu_component AS gpu_component

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NLLB_EXPECTED_COMPUTE_TYPE=float16

# CTranslate2 4.x is built against CUDA 12 and dlopen's libcublas.so.12 at
# inference time. The rest of the venv ships torch's CUDA 13 cuBLAS
# (libcublas.so.13), so GPU translation otherwise fails with
# "Library libcublas.so.12 is not found or cannot be loaded". Install the CUDA 12
# cuBLAS runtime and register it with the dynamic loader. cuBLAS alone is
# sufficient for NLLB translation; cuDNN is not needed and is omitted so it does
# not clash with torch's CUDA 13 cuDNN.
RUN "$COMPONENT_VIRTUALENV/bin/pip" install --no-cache-dir nvidia-cublas-cu12 \
    && echo "$COMPONENT_VIRTUALENV/lib/python3.12/site-packages/nvidia/cublas/lib" \
        > /etc/ld.so.conf.d/ctranslate2-cublas.conf \
    && ldconfig

FROM ${BUILD_TYPE}_component
