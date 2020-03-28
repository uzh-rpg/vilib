/*
 * Functions for handling preallocated memory-pools
 * memory_pool.cpp
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
