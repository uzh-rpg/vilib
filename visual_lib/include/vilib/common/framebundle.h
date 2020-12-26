/*
 * Combining multiple frames together for multicamera setups
 * framebundle.h
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

#include <memory>
#include <vector>
#include <mutex>
#include "vilib/common/frame.h"

namespace vilib {

class FrameBundle {
public:
  FrameBundle(void) = default;
  FrameBundle(const std::vector<std::shared_ptr<Frame>> & frames);
  ~FrameBundle(void) = default;

  inline std::size_t id(void) const {
    return bundle_id_;
  }

  inline bool empty(void) const {
    return frames_.empty();
  }

  inline std::size_t size(void) const {
    return frames_.size();
  }

  inline const std::shared_ptr<Frame> & at(std::size_t i) const {
    return frames_[i];
  }

  inline std::shared_ptr<Frame> & at(std::size_t i) {
    return frames_[i];
  }
private:
  static std::size_t getNewId(void);

  std::size_t bundle_id_;
  std::vector<std::shared_ptr<Frame>> frames_;

  static std::size_t last_id_;
  static std::mutex last_id_mutex_;
};

} // namespace vilib
