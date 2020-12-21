/*
 * FAST feature detector on the CPU (as provided by OpenCV)
 * fast_cpu.cpp
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

#include <assert.h>
#include <algorithm>
#include "vilib/feature_detection/fast/opencv/fast_cpu.h"

namespace vilib {
namespace opencv {

// Minimum border: 3, this is the minimum border needed because of the Bresenham circle
#define MINIMUM_BORDER                3

template<bool use_grid>
FASTCPU<use_grid>::FASTCPU(const std::size_t image_width,
                 const std::size_t image_height,
                 const std::size_t cell_size_width,
                 const std::size_t cell_size_height,
                 const std::size_t min_level,
                 const std::size_t max_level,
                 const std::size_t horizontal_border,
                 const std::size_t vertical_border,
                 const float threshold) :
  DetectorBase<use_grid>(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               std::max(static_cast<std::size_t>(MINIMUM_BORDER),horizontal_border),
               std::max(static_cast<std::size_t>(MINIMUM_BORDER),vertical_border)),
   threshold_(threshold),
   /*
    * Note to future self:
    * parameters: thresholds, nms, arc length
    * TYPE_9_16: 16 points, requires 9 in a row on a circle of 16
    */
   detector_(cv::FastFeatureDetector::create(
                               threshold_,
                               true,
                               cv::FastFeatureDetector::TYPE_9_16)) {
}

template<bool use_grid>
FASTCPU<use_grid>::~FASTCPU(void) {
}

template<bool use_grid>
void FASTCPU<use_grid>::detect(const std::vector<cv::Mat> & image_pyramid) {
  for(unsigned int l=this->min_level_;l<image_pyramid.size() && l<this->max_level_;++l) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image_pyramid[l],keypoints);

    const int maxw = image_pyramid[l].cols-this->horizontal_border_;
    const int maxh = image_pyramid[l].rows-this->vertical_border_;
    for(std::size_t i=0;i<keypoints.size();++i) {
      cv::KeyPoint & p = keypoints[i];
      if(p.pt.x < static_cast<int>(this->horizontal_border_) ||
         p.pt.y < static_cast<int>(this->vertical_border_) ||
         p.pt.x >= maxw || 
         p.pt.y >= maxh) {
        continue;
      }

      double scale = static_cast<double>(1<<l);
      double sub_x = p.pt.x*scale;
      double sub_y = p.pt.y*scale;
      double score = p.response;
      unsigned int level = l;
      this->addFeaturePoint(sub_x,sub_y,score,level);
    }
  }
}

// Explicit instantiations
template class FASTCPU<false>;
template class FASTCPU<true>;

} // namespace opencv
} // namespace vilib
