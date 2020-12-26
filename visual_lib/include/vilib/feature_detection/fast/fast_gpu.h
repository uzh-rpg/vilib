/*
 * FAST feature detector on the GPU
 * fast_gpu.h
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
#include "vilib/feature_detection/fast/fast_common.h"
#include "vilib/feature_detection/fast/fast_gpu_config.h"
#include "vilib/storage/subframe.h"

namespace vilib {

class FASTGPU : public DetectorBaseGPU {
public:
  FASTGPU(const std::size_t image_width,
          const std::size_t image_height,
          const std::size_t cell_size_width,
          const std::size_t cell_size_height,
          const std::size_t min_level,
          const std::size_t max_level,
          const std::size_t horizontal_border,
          const std::size_t vertical_border,
          const float threshold,
          const int min_arc_length,
          fast_score score);
  ~FASTGPU(void);
  void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid);
  void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid,
              std::function<void(const std::size_t & /* cell count */,
                                 const float *       /* pos */,
                                 const float *       /* score */,
                                 const int *         /* level */)> callback);
private:
  void detectBase(const std::vector<std::shared_ptr<Subframe>> & pyramid);

  // Detector parameters
  float threshold_;     // threshold value for creating a histerisis, usually 10.0f
  int min_arc_length_;  // minimum arc length for considering a point being a corner (minimum 9-10)
                        // Note: if the min_arc_length is < 9, then the GPU corner prechecks are skipped, as they are optimized for min. 9
  fast_score score_;
  // Derived parameters
  std::size_t det_horizontal_border_;
  std::size_t det_vertical_border_;

  // Temporary buffers
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  unsigned int * d_corner_lut_;
#else
  unsigned char * d_corner_lut_;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
};

} // namespace vilib
