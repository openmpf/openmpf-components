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

#include "Cluster.h"
#include "DetectionLocation.h"
#include "Track.h"

namespace MPF{
 namespace COMPONENT{

    /** ***************************************************************************
    *   add a single member to a cluster
    *
    * @tparam T     type of objects in cluster
    * @tparam fFunc member function of T to retrieve feature vector
    * @tparam dFunc function to compute distance between two feature vectors
    *
    * @param  m object to move into the cluster's member list
    *
    **************************************************************************** */
    template<class T, featureFunc<T> fFunc, distanceFunc dFunc>
    void Cluster<T,fFunc,dFunc>::add(T&& m){
      if(members.empty()){
        aveFeature = CALL_MEMBER_FUNC(m,fFunc)();
      }else{
        cv::normalize((aveFeature * members.size()) + (m.*fFunc)(), aveFeature,1.0,0.0,cv::NORM_L2);
      }
      members.push_back(move(m));
    }

    /** ***************************************************************************
    *  very naive agglomerate clustering,
    *  moves items from a list into cluster member lists in a list of clusters
    *
    * @tparam T     type of objects in cluster
    * @tparam fFunc member function of T to retrieve feature vector
    * @tparam dFunc function to compute distance between two feature vectors
    *
    * @param[in,out] lis     list of objects to be moved into clusters
    * @param[in,out] clList  list of clusters to move objects into
    * @param[in]     maxDist maximum feature distance an object can be from
    *                        its cluster's average feature
    *
    **************************************************************************** */
    template<class T, featureFunc<T> fFunc, distanceFunc dFunc>
    void Cluster<T,fFunc,dFunc>::cluster(list<T>                        &lis,
                                         list<Cluster<T, fFunc, dFunc>> &clList,
                                         float                           maxDist){
      auto lisItr = lis.begin();
      while(lisItr != lis.end()){

        // find 1st cluster *lisItr can join
        auto clItr = clList.begin();
        while(clItr != clList.end()){
          if(CALL_FUNC(dFunc)( clItr->aveFeature, CALL_MEMBER_FUNC(*lisItr,fFunc)() )  < maxDist) break;
          clItr++;
        }

        if(clItr != clList.end()){
          cv::normalize(  (clItr->aveFeature * clItr->members.size())
                         + CALL_MEMBER_FUNC(*lisItr,fFunc)(),
                        clItr->aveFeature,1.0,0.0,cv::NORM_L2);
          clItr->members.splice(clItr->members.end(),lis, lisItr++);           // found a cluster to join
        }else{
          clList.emplace_back();                                               // no cluster found, make a new one
          clList.back().aveFeature = CALL_MEMBER_FUNC(*lisItr,fFunc)();
          clList.back().members.splice(clList.back().members.end(),lis, lisItr++);
        }
      }
    }

    /** ***************************************************************************
    *  very naive agglomerative clustering,
    *  move items from a list list of clusters that is returned
    *
    * @tparam T     type of objects in cluster
    * @tparam fFunc member function of T to retrieve feature vector
    * @tparam dFunc function to compute distance between two feature vectors
    *
    * @param[in,out] lis     list of objects to be moved into clusters
    * @param[in]     maxDist maximum feature distance an object can be from
    *                        its cluster's average feature
    *
    * @returns  list of clusters that object have been move into
    *
    **************************************************************************** */
    template<class T, featureFunc<T> fFunc, distanceFunc dFunc>
    list<Cluster<T, fFunc, dFunc>> Cluster<T,fFunc,dFunc>::cluster(list<T> &lis,
                                                                   float    maxDist){
      list<Cluster<T, fFunc, dFunc>> clusterList;
      cluster(lis,clusterList,maxDist);
      return clusterList;
    }

    /** ***************************************************************************
    * Some explicit template instantiations used elsewhere in project
    * **************************************************************************** */
    template class Cluster<DetectionLocation, &DetectionLocation::getClassFeature,&cosDist>;
    template class Cluster<Track, &Track::getClassFeature,&cosDist>;

    /**/
 }
}
