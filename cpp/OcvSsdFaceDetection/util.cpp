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

namespace MPF{
  namespace COMPONENT{

    /** **************************************************************************
    * If a test rectange is within snapDist from frame edges then change another
    * rectangel to touch the corresponding frame edges
    *
    * \param   rt           rectangle to for snap distance test
    * \param   rm           rectangle to change and return
    * \param   frameSize    frame boundary size to snap to
    * \param   edgeSnapDist max distance (as fraction of size dimensions) from the edge for a snap
    *
    * \returns altered version of 'rm' rectangle that is touching the frame edges if snapped
    *          or the original rectangle 'rm' if no snap happened
    *
    *************************************************************************** */
    cv::Rect2i snapToEdges(const cv::Rect2i &rt,
                           const cv::Rect2i &rm,
                           const cv::Size2i &frameSize,
                           const float       edgeSnapDist=0.0075){

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

    /** ****************************************************************************
     *  print out opencv matrix on a single line
     *
     * \param   m matrix to serialize to single line string
     * \returns single line string representation of matrix
     *
    ***************************************************************************** */
    string format(cv::Mat1f m){
      stringstream ss;
      ss << "[";
      for(int r=0; r<m.rows; r++){
        for(int c=0; c<m.cols; c++){
          ss << setfill('0') << setw(6) << fixed << setprecision(3) << m.at<float>(r,c);
          if(c!=m.cols-1){
            ss << ", ";
          }else if(r!=m.rows-1){
            ss << "; ";
          }
        }
      }
      ss << "]";
      return ss.str();;
    }

    /** ****************************************************************************
     *  print out dlib matrix on a single line
     *
     * \param   m matrix to serialize to single line string
     * \returns single line string representation of matrix
     *
    ***************************************************************************** */
    template<typename T>
    string dformat(dlib::matrix<T> m){
      stringstream ss;
      ss << "{";
      for(size_t r=0;r<m.nr();r++){
        for(size_t c=0;c<m.nc();c++){
          ss << m(r,c);
          if(c != m.nc()-1) ss << ",";
        }
        if(r != m.nr()-1) ss << "; ";
      }
      ss << "}";
      string str = ss.str();
      return str;
    }
    // Some explicit template instantiations used elsewhere in project
    template string dformat<long>(dlib::matrix<long> m);

    /** **************************************************************************
    *   Parse a string into a vector
    *   e.g. [1,2,3,4]
    *************************************************************************** */
    template<typename T>
    vector<T> fromString(const string data){
      string::size_type begin = data.find('[') + 1;
      string::size_type end = data.find(']', begin);
      stringstream ss( data.substr(begin, end - begin) );
      vector<T> ret;
      T val;
      while(ss >> val){
        ret.push_back(val);
        ss.ignore(); // ignore seperating tokens e.g. ','
      }
      return ret;
    }
    // explicit template instantiation
    template vector<OrientationType> fromString<OrientationType>(const string data);

    /** **************************************************************************
    *   Parse a string into a opencv matrix
    *   e.g. [1,2,3,4, 5,6,7,8]
    *************************************************************************** */
    cv::Mat fromString(const string data,const int rows,const int cols,const string dt="f"){
      stringstream ss;
      ss << "{\"mat\":{\"type_id\":\"opencv-matrix\""
        << ",\"rows\":" << rows
        << ",\"cols\":" << cols
        << ",\"dt\":\"" << dt << '"'
        << ",\"data\":" << data << "}}";
      cv::FileStorage fs(ss.str(), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
      cv::Mat mat;
      fs["mat"] >> mat;
      return mat;
    }


    /** ****************************************************************************
    *   get MPF properties of various types
    *
    * \param T   data type to cast the result to
    * \param p   properties map to get property from
    * \param k   string key to use to retrieve property
    * \param def default value to return if key is not found
    *
    * \return  type converted value of property retrived with key or the default
    *
    ***************************************************************************** */
    template<typename T>
    T get(const Properties &p, const string &k, const T def){
      return DetectionComponentUtils::GetProperty<T>(p,k,def);
    }

    template int    get<int>   (const Properties&, const string&, const int);
    template bool   get<bool>  (const Properties&, const string&, const bool);
    template float  get<float> (const Properties&, const string&, const float);
    template string get<string>(const Properties&, const string&, const string);



    /** ****************************************************************************
    *   get configuration from environment variables if not
    *   provided by job configuration
    *
    * \param T   data type to cast the result to
    * \param p   properties map to get property from
    * \param k   string key to use to retrieve property
    * \param def default value to return if key is not found
    *
    * \return  type converted value of property retrived with key or the default
    *
    ***************************************************************************** */
    template<typename T>
    T getEnv(const Properties &p, const string &k, const T def){
      auto iter = p.find(k);
      if (iter == p.end()){
        const char* env_p = getenv(k.c_str());
        if(env_p != NULL){
          map<string,string> envp;
          envp.insert(pair<string,string>(k,string(env_p)));
          return DetectionComponentUtils::GetProperty<T>(envp,k,def);
        }else{
          return def;
        }
      }
      return DetectionComponentUtils::GetProperty<T>(p,k,def);
    }

    template int    getEnv<int>   (const Properties&, const string&, const int);
    template bool   getEnv<bool>  (const Properties&, const string&, const bool);
    template float  getEnv<float> (const Properties&, const string&, const float);
    template string getEnv<string>(const Properties&, const string&, const string);


    /** **************************************************************************
    *   Dump MPFLocation to a stream
    *************************************************************************** */
    ostream& operator<< (ostream& os, const MPFImageLocation& l) {
      os << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
                << l.width        << "," << l.height       << "):"
                << l.confidence;
      if(l.detection_properties.find("CLASSIFICATION") != l.detection_properties.end()){
        os << "|" << l.detection_properties.at("CLASSIFICATION");
      }

      return os;
    }

    /** **************************************************************************
    *   Dump MPFTrack to a stream
    *************************************************************************** */
    ostream& operator<< (ostream& os, const MPFVideoTrack& t) {
      os << t.start_frame << endl;
      os << t.stop_frame  << endl;
      for(auto& p:t.frame_locations){
        os << p.second.x_left_upper << "," << p.second.y_left_upper << ","
            << p.second.width << "," << p.second.height << endl;
      }
      return os;
    }

    /** ****************************************************************************
    *   Dump vectors to a stream
    ***************************************************************************** */
    template<typename T>
    ostream& operator<< (ostream& os, const vector<T>& v) {
      os << "{";
      size_t last = v.size() - 1;
      for(size_t i = 0; i < v.size(); ++i){
        os << v[i];
        if(i != last) os << ", ";
      }
      os << "}";
      return os;
    }

    // Explicit template instantiations
    template ostream& operator<< (ostream& os, const vector<int>&  v);
    template ostream& operator<< (ostream& os, const vector<long>& v);
    template ostream& operator<< (ostream& os, const vector<OrientationType>& v);

  }
}

/** **************************************************************************
*   Redefine ocv output for rect
*************************************************************************** */
std::ostream& cv::operator<< (std::ostream& os, const cv::Rect& r) {
  os << "[" << r.x << "," << r.y << "]-(" << r.width << "," << r.height << ")";
  return os;
}
