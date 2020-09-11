/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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
#include "util.h"

using namespace MPF::COMPONENT;

/** **************************************************************************
* If a test rectange is within snapDist from frame edges then change another
* rectangel to touch the corresponding frame edges
*
* \param   rt           rectangle to for snap distance test
* \param   rm           rectangle to change and return
* \param   frame        frame whose edges to snap to
* \param   edgeSnapDist max distance in pixels from the edge for a snap
*
* \returns altered 'rm' rectangle that is touching the frame edges if snapped
*          or the original rectangle 'rm' if no snap happened
*
*************************************************************************** */
cv::Rect2i MPF::COMPONENT::snapToEdges(const cv::Rect2i& rt, const cv::Rect2i& rm, const cv::Size2i& frameSize, const float edgeSnapDist=0.0075){

  cv::Point2i rt_tl = rt.tl();
  cv::Point2i rt_br = rt.br();
  cv::Point2i rm_tl = rm.tl();
  cv::Point2i rm_br = rm.br();

  int border_x = static_cast<int>(edgeSnapDist * frameSize.width);
  int border_y = static_cast<int>(edgeSnapDist * frameSize.height);

  if(rt_tl.x <= border_x){                             // near   left side of frame
    rm_tl.x = 0;
  }else if(rt_br.x >= frameSize.width - border_x - 1){ // near  right side of frame
    rm_br.x = frameSize.width - 1;
  }

  if(rt_tl.y <= border_y){                              // near    top side of frame
    rm_tl.y = 0;
  }else if(rt_br.y >= frameSize.height - border_y - 1){ // near bottom side of frame
    rm_br.y = frameSize.height - 1;
  }

  return cv::Rect2i(rm_tl , rm_br);
}