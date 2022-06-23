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

#ifndef OPENMPF_COMPONENTS_CLUSTER_H
#define OPENMPF_COMPONENTS_CLUSTER_H

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "util.h"


template<typename T>
class Cluster {
public:
    // average feature (centroid) of the cluster
    cv::Mat averageFeature;
    // members of the cluster
    std::vector<T> members;

    explicit Cluster(T member)
            : averageFeature(member.getClassFeature()) {
        members.push_back(std::move(member));
    }

    // These are needed to prevent a template error when Clusters are inserted into vectors.
    Cluster(Cluster &&) = default;

    Cluster(const Cluster &) = delete;


    /** ***************************************************************************
    *   add a single member to a cluster
    *
    * @tparam T  type of objects in cluster
    *
    * @param newMember  object to move into the cluster's member list
    *
    **************************************************************************** */
    void add(T newMember) {
        if (members.empty()) {
            averageFeature = newMember.getClassFeature();
        } else {
            cv::normalize((averageFeature * members.size()) + newMember.getClassFeature(),
                          averageFeature);
        }
        members.push_back(std::move(newMember));
    }
};


/** ***************************************************************************
*  very naive agglomerative clustering (not suitable for lots of items),
*  move items to a vector of clusters that is returned
*
* @tparam T              type of objects in cluster
* @tparam TDistanceFunc  function to compute distance between two feature vectors
*
* @param[in,out] items     vector of objects to be moved into clusters
* @param[in]     maxDist   maximum feature distance an object can be from
*                          its cluster's average feature
* @param[in] distanceFunc  function to compute distance between two feature vectors
*
* @returns vector of clusters that object have been moved into
*
**************************************************************************** */
template<typename T, typename TDistanceFunc = decltype(cosDist)>
std::vector<Cluster<T>> clusterItems(std::vector<T> items, float maxDist,
                                     TDistanceFunc &&distanceFunc = cosDist) {
    std::vector<Cluster<T>> clusters;
    for (auto &item: items) {
        auto feature = item.getClassFeature();
        // find 1st cluster item can join
        // TODO: Should we be using the minimum rather than the first?
        auto matchingCluster = std::find_if(
                clusters.begin(), clusters.end(),
                [&](const Cluster<T> &cluster) {
                    return distanceFunc(cluster.averageFeature, feature) <= maxDist;
                });
        if (matchingCluster != clusters.end()) {
            // found a cluster to join
            matchingCluster->add(std::move(item));
        } else {
            // no cluster found, make a new one
            clusters.emplace_back(std::move(item));
        }
    }
    return clusters;
}

#endif //OPENMPF_COMPONENTS_CLUSTER_H
