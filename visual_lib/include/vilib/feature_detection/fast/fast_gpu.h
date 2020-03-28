/*
 * FAST feature detector on the GPU
 * fast_gpu.h
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
