/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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


#ifndef OPENMPF_COMPONENTS_MODELFILEPARSEREXCEPTION_H
#define OPENMPF_COMPONENTS_MODELFILEPARSEREXCEPTION_H


#include <string.h>

class ModelFileParserException
{
public:

    ModelFileParserException(const char* str)
    {
        m_str = new char[strlen(str) + 1];
        strcpy(m_str, str);
    }

    ModelFileParserException(const ModelFileParserException &o)
    {
        m_str = new char[strlen(o.m_str) + 1];
        strcpy(m_str, o.m_str);
    }

    ~ModelFileParserException() {delete [] m_str;}

    const char* c_str() const {return m_str;}

private:

    char* m_str;
};


#endif //OPENMPF_COMPONENTS_MODELFILEPARSEREXCEPTION_H
