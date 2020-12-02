#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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

ARG MODELS_REGISTRY=openmpf/

FROM ${MODELS_REGISTRY}openmpf_trtis_detection_models:5.0 as models


FROM nvcr.io/nvidia/tensorrtserver:19.04-py3 as openmpf_trtis_server

COPY --from=models /models/model.graphdef /models/ip_irv2_coco/1/model.graphdef

# Specify custom models here.
# COPY <Path_to_custom_models>/models /models/
COPY models /models

COPY docker-entrypoint.sh /opt/tensorrtserver

ENTRYPOINT ["/opt/tensorrtserver/docker-entrypoint.sh"]

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRTIS Server." \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
