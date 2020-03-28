/*
 * FAST feature detector on the CPU (as provided by OpenCV)
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

#include <opencv2/features2d.hpp>
#include "vilib/feature_detection/detector_base.h"

namespace vilib {
namespace opencv { 

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
          const float threshold);
  ~FASTCPU(void);
  void detect(const std::vector<cv::Mat> & image) override;
private:
  float threshold_;
  cv::Ptr<cv::Feature2D> detector_;
};

} // namespace opencv
} // namespace vilib
