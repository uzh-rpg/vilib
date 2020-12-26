/*
 * Base class for feature detectors
 * detector_base.cpp
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

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_benchmark.h"

namespace vilib {

template<bool use_grid>
DetectorBase<use_grid>::DetectorBase(const std::size_t image_width,
                           const std::size_t image_height,
                           const std::size_t cell_size_width,
                           const std::size_t cell_size_height,
                           const std::size_t min_level,
                           const std::size_t max_level,
                           const std::size_t horizontal_border,
                           const std::size_t vertical_border) :
  cell_size_width_(cell_size_width),
  cell_size_height_(cell_size_height),
  n_cols_((image_width +cell_size_width -1)/cell_size_width),
  n_rows_((image_height+cell_size_height-1)/cell_size_height),
  min_level_(min_level),
  max_level_(max_level),
  horizontal_border_(horizontal_border),
  vertical_border_(vertical_border),
  // TODO: should be merged into grid
  grid_(cell_size_width,
        cell_size_height,
        n_cols_,
        n_rows_) {
  // populate keypoints_ grid cell for display!
  if(use_grid) {
    keypoints_.resize(n_cols_ * n_rows_, FeaturePoint(0.0,0.0,0.0,-1));
  }
}

template<bool use_grid>
void DetectorBase<use_grid>::detect(const std::vector<cv::Mat> & image) {
  (void)image;
}

template<bool use_grid>
void DetectorBase<use_grid>::reset(void) {
  BENCHMARK_START_HOST(DetectorBenchmark,Reset,false);
  if(use_grid) {
    grid_.reset();
  } else {
    keypoints_.clear();
  }
  BENCHMARK_STOP_HOST(DetectorBenchmark,Reset);
}

template<bool use_grid>
std::size_t DetectorBase<use_grid>::count(void) const {
  if(use_grid) {
    return grid_.getOccupiedCount();
  } else {
    return keypoints_.size();
  }
}

template<bool use_grid>
void DetectorBase<use_grid>::addFeaturePoint(double x, double y, double score, unsigned int level) {
  if(use_grid) {
    // check if we have already something in this cell?
    std::size_t cell_index = ((std::size_t)(y/cell_size_height_))*n_cols_ + ((std::size_t)(x/cell_size_width_));
    bool cell_occupied = grid_.isOccupied(cell_index);
    if(((cell_occupied == true) && keypoints_[cell_index].score_ < score) ||
      (cell_occupied == false)) {
      keypoints_[cell_index] = FeaturePoint(x,y,score,level);
      grid_.setOccupied(cell_index);
    }
  } else {
    keypoints_.emplace_back(x,y,score,level);
  }
}

template<bool use_grid>
void DetectorBase<use_grid>::displayFeatures(const char * title,
                                      const std::vector<cv::Mat> & image_pyramid,
                                      bool draw_on_level0,
                                      bool draw_cells) const {
  for(std::size_t l=0;l<image_pyramid.size();++l) {
    cv::Mat canvas;
    cv::cvtColor(image_pyramid[l],canvas,cv::COLOR_GRAY2RGB);
    // draw circles for the identified keypoints
    for(std::size_t i=0;i<keypoints_.size();++i) {
      // skip points whose grid cell is not occupied
      if (use_grid && grid_.isOccupied(i)==false){
        continue;
      } else if(!draw_on_level0 && keypoints_[i].level_ != l) {
        continue;
      }
      double scale = draw_on_level0?1.0:static_cast<double>(1<<l);
      double x = keypoints_[i].x_ * 1024 / scale;
      double y = keypoints_[i].y_ * 1024 / scale;
      cv::circle(canvas,
                 cv::Point(static_cast<int>(x),static_cast<int>(y)),
                 (keypoints_[i].level_+1)*3*1024,
                 cv::Scalar(0,0,255), // B,G,R
                 1,
                 8,
                 10);
    }
    if(use_grid && draw_on_level0 && draw_cells) {
      for(std::size_t r=0;r<n_rows_;++r) {
        for(std::size_t c=0;c<n_cols_;++c) {
          cv::rectangle(canvas,
                        cv::Point(c*cell_size_width_,r*cell_size_height_),
                        cv::Point((c+1)*cell_size_width_,(r+1)*cell_size_height_),
                        cv::Scalar(244,215,66), // B,G,R
                        1,
                        8,
                        0);
        }
      }
    }
    cv::imshow(title, canvas);
    cv::waitKey();

    if(draw_on_level0) {
      break;
    }
  }
}

// Explicit instantiations
template class DetectorBase<false>;
template class DetectorBase<true>;

} // namespace vilib
