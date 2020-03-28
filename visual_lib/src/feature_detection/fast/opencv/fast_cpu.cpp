/*
 * FAST feature detector on the CPU (as provided by OpenCV)
 * fast_cpu.cpp
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
#include <algorithm>
#include "vilib/feature_detection/fast/opencv/fast_cpu.h"

namespace vilib {
namespace opencv {

// Minimum border: 3, this is the minimum border needed because of the Bresenham circle
#define MINIMUM_BORDER                3

FASTCPU::FASTCPU(const std::size_t image_width,
                 const std::size_t image_height,
                 const std::size_t cell_size_width,
                 const std::size_t cell_size_height,
                 const std::size_t min_level,
                 const std::size_t max_level,
                 const std::size_t horizontal_border,
                 const std::size_t vertical_border,
                 const float threshold) :
  DetectorBase(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               std::max((std::size_t)MINIMUM_BORDER,horizontal_border),
               std::max((std::size_t)MINIMUM_BORDER,vertical_border)),
   threshold_(threshold),
   detector_(cv::FastFeatureDetector::create(
                               threshold_,
                               true,
                               cv::FastFeatureDetector::TYPE_9_16)) {
  /*
   * Note to future self:
   * parameters: thresholds, nms, arc length
   * TYPE_9_16: 16 points, requires 9 in a row on a circle of 16
   */
}

FASTCPU::~FASTCPU(void) {
}

void FASTCPU::detect(const std::vector<cv::Mat> & image_pyramid) {
  for(unsigned int l=min_level_;l<image_pyramid.size() && l<max_level_;++l) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image_pyramid[l],keypoints);

    const int maxw = image_pyramid[l].cols-horizontal_border_;
    const int maxh = image_pyramid[l].rows-vertical_border_;
    for(std::size_t i=0;i<keypoints.size();++i) {
      cv::KeyPoint & p = keypoints[i];
      if(p.pt.x < (int)horizontal_border_ ||
         p.pt.y < (int)vertical_border_ ||
         p.pt.x >= maxw || 
         p.pt.y >= maxh) {
        continue;
      }

      double scale = (double)(1<<l);
      double sub_x = p.pt.x*scale;
      double sub_y = p.pt.y*scale;
      double score = p.response;
      unsigned int level = l;

      addFeaturePoint(sub_x,sub_y,score,level);
    }
  }
}

} // namespace opencv
} // namespace vilib
