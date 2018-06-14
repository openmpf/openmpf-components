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


#ifndef SPSCBOUNDEDQUEUE_H
#define SPSCBOUNDEDQUEUE_H
#include <atomic>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>

template <typename T>
class SPSCBoundedQueue {
 private:
  std::atomic<int> head{0};
  std::atomic<int> tail{0};
  int capacity;
  std::vector<T> buffer;

 public:

  SPSCBoundedQueue() = delete;
  SPSCBoundedQueue(uint32_t c) 
      : capacity(c), buffer(c) {}

  bool isEmpty() {
    int current_tail = std::atomic_load_explicit(&tail, std::memory_order_relaxed);
    int current_head = std::atomic_load_explicit(&head, std::memory_order_acquire);
    return (current_head == current_tail);
  }

  bool isFull() {
    int current_tail = std::atomic_load_explicit(&tail, std::memory_order_relaxed);
    int current_head = std::atomic_load_explicit(&head, std::memory_order_acquire);
    return ((current_head+1)%capacity == current_tail);
  }

  bool push(T &entry) {
    // The tail is read-write here, but this function is the only
    // place where it is modified, and since there is only one
    // producer thread, we can use relaxed memory order to read
    // it, but must use release memory order to write it.
    // We want to use acquire memory order here to read the head
    // to ensure proper memory ordering with respect to the write
    // by the consumer.
    int current_tail = std::atomic_load_explicit(&tail, std::memory_order_relaxed);
    int current_head = std::atomic_load_explicit(&head, std::memory_order_acquire);
    // std::cout << "push: tail = " << current_tail << " head = " << current_head << std::endl;
    // No space available. Try again later.
    if ((current_head+1)%capacity == current_tail) {
      return false;
    }
    else {
      // Move the entry into the buffer. Increment the tail index.
      buffer[current_tail] = std::move(entry);
      std::atomic_store_explicit(&tail, (current_tail+1)%capacity, std::memory_order_release);
      return true;
    }
  }

  bool pop(T &entry) {
    int current_head;
    int current_tail;
    // The head is read-write here, but this function is the only
    // place where it is modified, and since there is only one
    // consumer thread, we can use relaxed memory order to read
    // it, but must use release memory order to write it.
    // We want to use acquire memory order here to read the tail
    // to ensure proper memory ordering with respect to the write
    // by the producer.
    current_head = std::atomic_load_explicit(&head, std::memory_order_relaxed);
    current_tail = std::atomic_load_explicit(&tail, std::memory_order_acquire);
    // std::cout << "pop: head = " << current_head << " tail = " << current_tail << std::endl;
    if (current_head == current_tail) {
      // Queue is empty. Try again later.
      return false;
    }
    else {
      // Return a reference to the head entry in the queue, and
      // increment the head index.
      entry = buffer[current_head];
      std::atomic_store_explicit(&head, (current_head+1)%capacity, std::memory_order_release);
      return true;
    }
  }

};


struct VideoFrame {
  int index;
  cv::Mat frame;

  VideoFrame() : index(0) {}
  VideoFrame(int i, cv::Mat &f) 
      : index(i), frame(f) {}
};



#endif
