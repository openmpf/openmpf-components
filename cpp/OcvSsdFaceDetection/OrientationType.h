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

#include <typeinfo>
#include <istream>
#include <ostream>
#include <opencv2/core/types.hpp>

namespace MPF{
  namespace COMPONENT{

    using namespace std;
    enum OrientationType {ROTATE_90_CLOCKWISE        = cv::ROTATE_90_CLOCKWISE,
                          ROTATE_180                 = cv::ROTATE_180,
                          ROTATE_90_COUNTERCLOCKWISE = cv::ROTATE_90_COUNTERCLOCKWISE,
                          ROTATE_0};

    typedef vector<OrientationType>       OrientVec;                ///< vector of 90 Orientations

    /** ****************************************************************************
     *   Write OrientationType to ints
     ****************************************************************************** */
    inline
    ostream& operator<< (ostream& os, const OrientationType& t) {
      switch(t){
        case ROTATE_0:                   os <<   "0"; break;
        case ROTATE_180:                 os << "180"; break;
        case ROTATE_90_CLOCKWISE:        os << "270"; break;
        case ROTATE_90_COUNTERCLOCKWISE: os <<  "90"; break;
      }
      return os;
    }

    /** ****************************************************************************
     *  Convet OrientationType to CCW angle from vertical [0-360)
     *
     * \param orientation enum to convert to float
     *
     * \return floatingpoint value corresponding to angle
     *
     ****************************************************************************** */
    inline
    float degCCWfromVertical(const OrientationType& orientation){
        switch(orientation){
          case ROTATE_0:                   return   0.0;
          case ROTATE_90_CLOCKWISE:        return  90.0;
          case ROTATE_180:                 return 180.0;
          case ROTATE_90_COUNTERCLOCKWISE: return 270.0;
        }
    }

    /** ****************************************************************************
     *  Calculate CCW angle from vector as measued from vertical in range of [0,360)
     *
     * \param vec vector making the angle with the vertical axis
     *
     * \return ccw angle mad with vertical axis [0,360)
     *
     ****************************************************************************** */
     inline
     float degCCWfromVertical(const cv::Point2f vec){
      float angle = -90.0 - atan2(vec.y,vec.x) * 180.0 / M_PI;
      if(angle < 0.0){ angle += 360.0;}
      return angle;
     }

    /** ****************************************************************************
     *  Calculate angle difference between two angles a,b
     *
     * \param a 'a' angle in a - b
     * \param b 'b' angle in a - b
     *
     * \return a - b in range [-180,180)
     *
     ****************************************************************************** */
    inline
    int angleDiff(int a, int b){
      return (a - b + 540) % 360 - 180;
    }

    /** ****************************************************************************
     * Read OrientationType from ints in a stream
     ****************************************************************************** */
    inline
    istream& operator>> (istream& is, OrientationType& t ){
      unsigned int x;
      if(is >> x){
        switch(x){
        case   0: t = ROTATE_0;                   break;
        case  90: t = ROTATE_90_COUNTERCLOCKWISE; break;
        case 180: t = ROTATE_180;                 break;
        case 270: t = ROTATE_90_CLOCKWISE;        break;
        default:  throw new invalid_argument("Value " + to_string(x) + " of enum OrientationType is not supported");
        }
      }
      return is;
    }

    /** ****************************************************************************
     * Inverse transfrom
     *
     * \param orientation for which to get the inverse operation
     *
     * \return operation to undo rotation
     *
     ****************************************************************************** */
    inline
    OrientationType inv(const OrientationType& orientation){
      switch(orientation){
        case ROTATE_0:                    return ROTATE_0;
        case ROTATE_90_COUNTERCLOCKWISE:  return ROTATE_90_CLOCKWISE;
        case ROTATE_180:                  return ROTATE_180;
        case ROTATE_90_CLOCKWISE:         return ROTATE_90_COUNTERCLOCKWISE;
      }
    }


    /** ****************************************************************************
     * Rotate a point to in an image to a corresponding position if the image
     * were rotated to the given orientation
     *
     * \param pt           point to be rotated
     * \param orientation  orientation of image to which to rotate the point to
     * \param canvasSize   size of oriented/rotated destination image
     *
     * \return equivalent coordinates in rotated/oriented image
     *
    ***************************************************************************** */
    template<typename T> inline
    cv::Point_<T> rotate(const cv::Point_<T> &pt,const OrientationType &orientation, const cv::Size_<T> &canvasSize){
      cv::Point_<T> p(pt);
      switch(orientation) {
        case ROTATE_90_COUNTERCLOCKWISE: swap(p.x, p.y); p.y = canvasSize.height - p.y; break;
        case ROTATE_90_CLOCKWISE:        swap(p.x, p.y); p.x = canvasSize.width  - p.x; break;
        case ROTATE_180:   p.x = canvasSize.width - p.x; p.y = canvasSize.height - p.y; break;
      }
      return p;
    }

    /** ****************************************************************************
     * Rotate a rectangel to in an image to a corresponding position if the image
     * were rotated to the given orientation
     *
     * \param rec          rectangle for which to find corresponding recangle
     * \param orientation  orientation of image to which to rotate the point to
     * \param canvasSize   size of oriented/rotated destination image
     *
     * \return equivalent rectangle in rotated/oriented image
     *
    ***************************************************************************** */
    template<typename T> inline
    cv::Rect_<T> rotate(const cv::Rect_<T> &rec, const OrientationType &orientation, const cv::Size_<T> &canvasSize){
      cv::Point_<T> p1 = rotate(rec.tl(),orientation,canvasSize);
      cv::Point_<T> p2 = rotate(rec.br(),orientation,canvasSize);
      return cv::Rect_<T>(min(p1.x,p2.x),min(p1.y,p2.y),abs<T>(p1.x-p2.x),abs<T>(p1.y-p2.y));
    }

    /** ****************************************************************************
     * Rotate an image to the specified orientation
     *
     * \param img         image to be rotated
     * \param orientation orientation to rotate image to
     *
     * \return rotated image
     *
    ***************************************************************************** */
    inline
    cv::Mat rotate(cv::Mat img,const OrientationType &orientation){
      cv::Mat rot;
      switch(orientation){
        case ROTATE_0:                    rot = img;                                          break;
        case ROTATE_90_COUNTERCLOCKWISE:  cv::rotate(img,rot,cv::ROTATE_90_COUNTERCLOCKWISE); break;
        case ROTATE_180:                  cv::rotate(img,rot,cv::ROTATE_180);                 break;
        case ROTATE_90_CLOCKWISE:         cv::rotate(img,rot,cv::ROTATE_90_CLOCKWISE);        break;
      }
      return rot;
    }
  }
}