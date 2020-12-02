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


import setuptools

setuptools.setup(
    name='AzureFormDetection',
    version='5.1',
    packages=setuptools.find_packages(exclude=('*test*',)),
    package_data={'': ['text-tags.json']},
    install_requires=(
        'mpf_component_api>=5.1',
        'mpf_component_util>=5.1',
        'opencv-python>=3.3',
        'numpy>=1.11'
    ),
    entry_points={
        'mpf.exported_component': 'component = acs_form_detection_component.acs_form_detection_component:AcsFormDetectionComponent'
    }
)
