/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2016 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2016 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/


#ifndef OPENMPF_COMPONENTS_MODELFILEPARSER_H
#define OPENMPF_COMPONENTS_MODELFILEPARSER_H


#include <config4cpp/Configuration.h>
#include "ModelFileParserException.h"

class ModelFileParser
{
public:

    ModelFileParser();
    ~ModelFileParser();

    void parse(const char* model_filename, const char* scope)
    throw (ModelFileParserException);
    void listModelScopes(CONFIG4CPP_NAMESPACE::StringVector &vec);
    const char* getName(const char* model_scope)
    throw (ModelFileParserException);
    const char* getModelTxt(const char* model_cope)
    throw (ModelFileParserException);
    const char* getModelBin(const char* model_scope)
    throw (ModelFileParserException);
    const char* getSynsetTxt(const char* model_scope)
    throw (ModelFileParserException);

private:

    CONFIG4CPP_NAMESPACE::Configuration* m_cfg;
    CONFIG4CPP_NAMESPACE::StringBuffer m_scope;
    bool m_parseCalled;
    CONFIG4CPP_NAMESPACE::StringVector m_modelScopeNames;

};


#endif //OPENMPF_COMPONENTS_MODELFILEPARSER_H
