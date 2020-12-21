/*
 * Harris/Shi-Tomasi feature detector on the GPU
 * harris_gpu.h
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

#include <vector>
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/storage/subframe.h"
#include "vilib/preprocess/conv_filter.h"

namespace vilib {

class HarrisGPU : public DetectorBaseGPU {
public:
  HarrisGPU(const std::size_t image_width,
          const std::size_t image_height,
          const std::size_t cell_size_width,
          const std::size_t cell_size_height,
          const std::size_t min_level,
          const std::size_t max_level,
          const std::size_t horizontal_border,
          const std::size_t vertical_border,
          const conv_filter_border_type_t filter_border_type,
          // Use Harris (true) or Shi-Tomasi (false)
          const bool use_harris,
          // Harris constant, unused for Shi-Tomasi
          const float harris_k,
          // Features are dropped whose score is less than
          // (best feature score) * quality_level
          const float quality_level);
  ~HarrisGPU(void);
  void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid);
  void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid,
              std::function<void(const std::size_t & /* cell count */,
                                 const float *       /* pos */,
                                 const float *       /* score */,
                                 const int *         /* level */)> callback);
private:
  void detectBase(const std::vector<std::shared_ptr<Subframe>> & pyramid);

  // Detector parameters
  conv_filter_border_type_t filter_border_type_;
  float quality_level_; // Harris, or Shi-Tomasi quality level
  bool use_harris_; // Use Harris, or Shi-Tomasi as response score
  float harris_k_;  // Harris coefficient k, usually around ~ 0.04f

  // Intermediate buffers
  std::vector<float *> d_result_;
  std::vector<std::size_t> d_result_pitch_elements_;
};

} // namespace vilib
