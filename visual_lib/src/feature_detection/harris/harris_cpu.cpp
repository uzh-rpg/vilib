/*
 * Harris/Shi-Tomasi feature detector on the CPU (as provided by OpenCV)
 * harris_cpu.cpp
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

#include <opencv2/imgproc.hpp>
#include "vilib/feature_detection/harris/harris_cpu.h"

namespace vilib {
namespace opencv {

// neighborhood size (blocksize x blocksize)
#define BLOCK_SIZE                    3

template<bool use_grid>
HarrisCPU<use_grid>::HarrisCPU(const std::size_t image_width,
                     const std::size_t image_height,
                     const std::size_t cell_size_width,
                     const std::size_t cell_size_height,
                     const std::size_t min_level,
                     const std::size_t max_level,
                     const std::size_t horizontal_border,
                     const std::size_t vertical_border,
                     const bool use_harris,
                     const double harris_k,
                     const double quality_level,
                     const double min_euclidian_distance):
  DetectorBase<use_grid>(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               horizontal_border,
               vertical_border),
  use_harris_(use_harris),
  harris_k_(harris_k),
  quality_level_(quality_level),
  min_euclidean_distance_(min_euclidian_distance) {
  // To make it comparable with the cell grid version, we use the same number of corners
  // as the number of cells
  max_corner_count_ = use_grid ? 0 : this->n_cols_ * this->n_rows_;
}

template<bool use_grid>
HarrisCPU<use_grid>::~HarrisCPU(void) {
}

template<bool use_grid>
void HarrisCPU<use_grid>::detect(const std::vector<cv::Mat> & image_pyramid) {
  for(unsigned int l=this->min_level_;l<image_pyramid.size() && l< this->max_level_;++l) {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image_pyramid[l],
                            corners,
                            max_corner_count_,
                            quality_level_,
                            min_euclidean_distance_,
                            cv::Mat(),
                            BLOCK_SIZE,
                            use_harris_,
                            harris_k_);

    const int maxw = image_pyramid[l].cols-this->horizontal_border_;
    const int maxh = image_pyramid[l].rows-this->vertical_border_;
    for(std::size_t i=0;i<corners.size();++i) {
      cv::Point2f & p = corners[i];
      if(p.x < this->horizontal_border_ ||
         p.y < this->vertical_border_ ||
         p.x >= maxw ||
         p.y >= maxh) {
        continue;
      }

      double scale = static_cast<double>(1<<l);
      double sub_x = p.x*scale;
      double sub_y = p.y*scale;
      /*
       * Note to future self and others:
       * Unfortunately we do not get scores from goodFeaturesToTrack(), 
       * so we assign a score ourselves. In the multi-level detection,
       * this might cause problems.
       */
      double score = 1.0;
      this->addFeaturePoint(sub_x,sub_y,score,l);
    }
  }
}

// Explicit instantiations
template class HarrisCPU<false>;
template class HarrisCPU<true>;

} // namespace opencv
} // namespace vilib
