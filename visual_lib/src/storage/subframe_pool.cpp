/*
 * Functions for handling preallocated memory-pools
 * memory_pool.cpp
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <iostream>
#include "vilib/storage/subframe_pool.h"

namespace vilib {

SubframePool::SubframePool(std::size_t preallocated_item_num,
                           std::size_t width,
                           std::size_t height,
                           std::size_t data_bytes,
                           Subframe::MemoryType type) :
  items_size_(preallocated_item_num),
  width_(width),height_(height),data_bytes_(data_bytes),type_(type) {
  // preallocate a few subframes
  std::lock_guard<std::mutex> lock(items_mutex_);
  for(std::size_t i=0;i<preallocated_item_num;++i) {
    items_.push_front(std::shared_ptr<Subframe>(new Subframe(width,
                                                            height,
                                                            data_bytes,
                                                            type)));
    if(i == 0) {
      pitch_ = items_.front()->pitch_;
    }
  }
}

SubframePool::~SubframePool(void) {
}

std::shared_ptr<Subframe> SubframePool::get_subframe(void) {
  // Do we have any more items preallocated?
  if(items_.empty() == false) {
    std::lock_guard<std::mutex> lock(items_mutex_);
    // Yes, take the first element from the unused list, and return it
    std::shared_ptr<Subframe> ptr = items_.front();
    items_.pop_front();
    --items_size_;
    return ptr;
  }
  // No, just allocate a new element
  return std::shared_ptr<Subframe>(new Subframe(width_,
                                                height_,
                                                data_bytes_,
                                                type_));
}

void SubframePool::return_subframe(std::shared_ptr<Subframe> frame) {
  assert(frame->width_  == width_);
  assert(frame->height_ == height_);
  assert(frame->type_   == type_);
  std::lock_guard<std::mutex> lock(items_mutex_);
  items_.push_front(frame);
  ++items_size_;
}

std::size_t SubframePool::preallocated_num(void) {
  std::lock_guard<std::mutex> lock(items_mutex_);
  return items_size_;
}

} // namespace vilib
