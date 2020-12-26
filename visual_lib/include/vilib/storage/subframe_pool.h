/*
 * Functions for handling preallocated memory-pools
 * subframe_pool.h
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

#pragma once

#include <mutex>
#include <memory>
#include <forward_list>
#include "vilib/storage/subframe.h"

namespace vilib {

class SubframePool {
public:
  SubframePool(std::size_t preallocated_item_num,
               std::size_t width,
               std::size_t height,
               std::size_t data_bytes,
               Subframe::MemoryType type);
  ~SubframePool(void);
  std::shared_ptr<Subframe> get_subframe(void);
  void return_subframe(std::shared_ptr<Subframe> frame);
  std::size_t preallocated_num(void);
  inline std::size_t get_width(void) const  { return width_; }
  inline std::size_t get_height(void) const { return height_; }
  inline std::size_t get_pitch(void) const  { return pitch_; }
private:
  std::forward_list<std::shared_ptr<Subframe>> items_;
  std::size_t items_size_;
  std::mutex items_mutex_;
  std::size_t width_;
  std::size_t height_;
  std::size_t pitch_;
  std::size_t data_bytes_;
  Subframe::MemoryType type_;
};

} // namespace vilib
