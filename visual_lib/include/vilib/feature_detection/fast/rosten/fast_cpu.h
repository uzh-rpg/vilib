/*
 * FAST feature detector on the CPU (as provided by Edward Rosten)
 * fast_cpu.h
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

#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/fast/fast_common.h"

namespace vilib {
namespace rosten {

struct xys_tuple;

template <bool use_grid>
class FASTCPU : public DetectorBase<use_grid> {
public:
  FASTCPU(const std::size_t image_width,
          const std::size_t image_height,
          const std::size_t cell_size_width,
          const std::size_t cell_size_height,
          const std::size_t min_level,
          const std::size_t max_level,
          const std::size_t horizontal_border,
          const std::size_t vertical_border,
          const float threshold,
          const int min_arc_length,
          const fast_score score);
  ~FASTCPU(void);
  void detect(const std::vector<cv::Mat> & image) override;
private:
  float threshold_;
  struct xys_tuple * (*fn_)(const unsigned char * im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
};

} // namespace rosten
} // namespace vilib
