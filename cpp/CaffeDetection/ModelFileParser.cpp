/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2017 The MITRE Corporation                                       *
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

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <config4cpp/SchemaValidator.h>
#include "ModelFileParser.h"

using CONFIG4CPP_NAMESPACE::StringBuffer;
using CONFIG4CPP_NAMESPACE::StringVector;
using CONFIG4CPP_NAMESPACE::Configuration;
using CONFIG4CPP_NAMESPACE::ConfigurationException;
using CONFIG4CPP_NAMESPACE::SchemaValidator;

//-----------------------------------------------------------------------------
ModelFileParser::ModelFileParser()
{
    m_cfg = 0;
    m_parseCalled = false; // set to 'true' after a successful parse().
}

//-----------------------------------------------------------------------------
ModelFileParser::~ModelFileParser()
{
    m_cfg->destroy();
}

//-----------------------------------------------------------------------------
void ModelFileParser::parse(const char* model_filename, const char* scope)
throw (ModelFileParserException)
{
    SchemaValidator sv;
    StringBuffer filter;
    const char* schema[] = {
            "uid-model = scope",
            "uid-model.name = string",
            "uid-model.model_txt = string",
            "uid-model.model_bin = string",
            "uid-model.synset_txt = string",
            0
    };

    assert(!m_parseCalled);
    m_cfg = Configuration::create();
    m_scope = scope;
    Configuration::mergeNames(scope, "uid-model", filter);
    try {
        m_cfg->parse(model_filename);
        sv.parseSchema(schema);
        sv.validate(m_cfg, m_scope.c_str(), "");
        m_cfg->listFullyScopedNames(
                m_scope.c_str(),
                "",
                Configuration::CFG_SCOPE,
                false,
                filter.c_str(),
                m_modelScopeNames);
    } catch(const ConfigurationException & ex) {
        throw ModelFileParserException(ex.c_str());
    }
    m_parseCalled = true;
}

//-----------------------------------------------------------------------------
void ModelFileParser::listModelScopes(StringVector & vec)
{
    assert(m_parseCalled);
    vec = m_modelScopeNames;
}

//-----------------------------------------------------------------------------
const char* ModelFileParser::getName(const char* model_scope)
throw (ModelFileParserException)
{
    assert(m_parseCalled);
    try {
        return m_cfg->lookupString(model_scope, "name");
    } catch(const ConfigurationException & ex) {
        throw ModelFileParserException(ex.c_str());
    }
}

//-----------------------------------------------------------------------------
const char* ModelFileParser::getModelTxt(const char* model_scope)
throw (ModelFileParserException)
{
    assert(m_parseCalled);
    try {
        return m_cfg->lookupString(model_scope, "model_txt");
    } catch(const ConfigurationException & ex) {
        throw ModelFileParserException(ex.c_str());
    }
}

//-----------------------------------------------------------------------------
const char* ModelFileParser::getModelBin(const char* model_scope)
throw (ModelFileParserException)
{
    assert(m_parseCalled);
    try {
        return m_cfg->lookupString(model_scope, "model_bin");
    } catch(const ConfigurationException & ex) {
        throw ModelFileParserException(ex.c_str());
    }
}

//-----------------------------------------------------------------------------
const char* ModelFileParser::getSynsetTxt(const char* model_scope)
throw (ModelFileParserException)
{
    assert(m_parseCalled);
    try {
        return m_cfg->lookupString(model_scope, "synset_txt");
    } catch(const ConfigurationException & ex) {
        throw ModelFileParserException(ex.c_str());
    }
}


