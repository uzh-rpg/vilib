/*
 * Harris/Shi-Tomasi feature detector on the CPU (as provided by OpenCV)
 * harris_cpu.h
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

#include <opencv2/features2d.hpp>
#include "vilib/feature_detection/detector_base.h"

namespace vilib {
namespace opencv { 

template<bool use_grid>
class HarrisCPU : public DetectorBase<use_grid> {
public:
  HarrisCPU(const std::size_t image_width,
            const std::size_t image_height,
            const std::size_t cell_size_width,
            const std::size_t cell_size_height,
            const std::size_t min_level,
            const std::size_t max_level,
            const std::size_t horizontal_border,
            const std::size_t vertical_border,
            // Use Harris (true) or Shi-Tomasi (false)
            const bool use_harris,
            // Harris constant, unused for Shi-Tomasi
            const double harris_k,
            // Features are dropped whose score is less than
            // (best feature score) * quality_level
            const double quality_level,
            // Minimum Euclidiean distance between features
            const double min_euclidiean_distance);
  ~HarrisCPU(void);
  void detect(const std::vector<cv::Mat> & image) override;
private:
  bool use_harris_;
  const double harris_k_;
  const double quality_level_;
  const double min_euclidean_distance_;
  std::size_t max_corner_count_;
};

} // namespace opencv
} // namespace vilib
