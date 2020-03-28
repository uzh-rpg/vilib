/*
 * FAST feature detector on the CPU (as provided by Edward Rosten)
 * fast_cpu.h
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

#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/fast/fast_common.h"

namespace vilib {
namespace rosten {

struct xys_tuple;

template <bool use_grid>
class FASTCPU : public DetectorBase {
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
  std::size_t count(void) const override;
  void reset(void);
private:
  float threshold_;
  struct xys_tuple * (*fn_)(const unsigned char * im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
};

} // namespace rosten
} // namespace vilib
