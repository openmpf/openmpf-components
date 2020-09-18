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
#ifndef OCVYOLODETECTION_CLUSTER_H
#define OCVYOLODETECTION_CLUSTER_H

#include "types.h"

namespace MPF{
 namespace COMPONENT{

    using namespace std;

    template<class T>
    using featureFunc  = const cv::Mat&(T::*)() const;             // could parameterize away cv::Mat
    using distanceFunc = float(*)(const cv::Mat&, const cv::Mat&);

    template<class T, featureFunc<T> fFunc, distanceFunc dFunc>
    class Cluster{
      public:
        cv::Mat aveFeature;                            ///< average feature (centroid) of the cluster
        list<T> members;                               ///< members of the cluster

        void add(T&& m);                               ///< move an item into member list

        static void cluster(list<T>                           &lis,
                            list<Cluster<T, fFunc, dFunc>>    &clList,
                            float                              maxDist);  ///< cluster items in a list

        static list<Cluster<T, fFunc, dFunc>> cluster(list<T> &lis,
                                                      float    maxDist);  ///< cluster items in a list

   };
 }
}

#endif