/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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
#ifndef OPENMPF_COMPONENTS_TRITON_EXCEPTION_MACROS_H
#define OPENMPF_COMPONENTS_TRITON_EXCEPTION_MACROS_H

/** ****************************************************************************
* Macro for throwing exception so we can see where in the code it happened
***************************************************************************** */
#ifdef DEBUG_LINE_NUMBERS
#define THROW_TRITON_EXCEPTION(X, MSG) {                                       \
    MPF::COMPONENT::MPFDetectionError err = (X);                               \
    throw MPF::COMPONENT::MPFDetectionException(err,                           \
       "Error in " + std::string(__FILENAME__)                                 \
        + "[" + std::to_string(__LINE__) + "]: " + (MSG));                     \
}
#else
#define THROW_TRITON_EXCEPTION(X, MSG) {                                       \
    MPF::COMPONENT::MPFDetectionError err = (X);                               \
    throw MPF::COMPONENT::MPFDetectionException(err, (MSG));                   \
}
#endif

/** ****************************************************************************
* Macro for error checking / logging of inference server client lib
***************************************************************************** */
// TODO: Don't use MPF_OTHER_DETECTION_ERROR_TYPE.
#ifdef DEBUG_LINE_NUMBERS
#define TR_CHECK_OK(X, MSG) {                                                  \
  triton::client::Error err = (X);                                             \
  if (!err.IsOk()) {                                                           \
    throw MPF::COMPONENT::MPFDetectionException(MPF_OTHER_DETECTION_ERROR_TYPE \
      , std::string("Triton inference server error")                           \
        + " in " + std::string(__FILENAME__)                                   \
        + "[" + std::to_string( __LINE__) + "]"                                \
        + ": " + (MSG) + ": " + err.Message());                                \
  }                                                                            \
}
#else
#define TR_CHECK_OK(X, MSG) {                                                  \
  triton::client::Error err = (X);                                             \
  if (!err.IsOk()) {                                                           \
    throw MPF::COMPONENT::MPFDetectionException(MPF_OTHER_DETECTION_ERROR_TYPE \
      , std::string("Triton inference server error: ")                         \
        + (MSG) + ": " + err.Message());                                       \
  }                                                                            \
}
#endif

#endif // OPENMPF_COMPONENTS_TRITON_EXCEPTION_MACROS_H
