/*
 * Base class for feature detectors
 * detector_base.cpp
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

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_benchmark.h"

namespace vilib {

DetectorBase::DetectorBase(const std::size_t image_width,
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
  keypoints_.resize(n_cols_ * n_rows_, FeaturePoint(0.0,0.0,0.0,-1));
}

void DetectorBase::detect(const std::vector<cv::Mat> & image) {
  (void)image;
}

void DetectorBase::reset(void) {
  BENCHMARK_START_HOST(DetectorBenchmark,Reset,false);
  grid_.reset();
  BENCHMARK_STOP_HOST(DetectorBenchmark,Reset);
}

std::size_t DetectorBase::count(void) const {
  return grid_.getOccupiedCount();
}

void DetectorBase::addFeaturePoint(double x, double y, double score, unsigned int level) {
  // check if we have already something in this cell?
  std::size_t cell_index = ((std::size_t)(y/cell_size_height_))*n_cols_ + ((std::size_t)(x/cell_size_width_));
  bool cell_occupied = grid_.isOccupied(cell_index);
  if(((cell_occupied == true) && keypoints_[cell_index].score_ < score) ||
     (cell_occupied == false)) {
    keypoints_[cell_index] = FeaturePoint(x,y,score,level);
    grid_.setOccupied(cell_index);
  }
}

void DetectorBase::displayFeatureGrid(const char * title,
                                      const std::vector<cv::Mat> & image_pyramid,
                                      bool draw_on_level0,
                                      bool draw_cells) const {
  for(std::size_t l=0;l<image_pyramid.size();++l) {
    cv::Mat canvas;
    cv::cvtColor(image_pyramid[l],canvas,cv::COLOR_GRAY2RGB);
    // draw circles for the identified keypoints
    for(std::size_t i=0;i<keypoints_.size();++i) {
      // skip points whose grid cell is not occupied
      if (grid_.isOccupied(i)==false){
        continue;
      } else if(!draw_on_level0 && keypoints_[i].level_ != l) {
        continue;
      }
      double scale = draw_on_level0?1.0:(double)(1<<l);
      double x = keypoints_[i].x_ * 1024 / scale;
      double y = keypoints_[i].y_ * 1024 / scale;
      cv::circle(canvas,
                 cv::Point((int)x,(int)y),
                 (keypoints_[i].level_+1)*3*1024,
                 cv::Scalar(0,0,255), // B,G,R
                 1,
                 8,
                 10);
    }
    if(draw_on_level0 && draw_cells) {
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

void DetectorBase::displayFeatures(const char * title,
                                   const std::vector<cv::Mat> & image_pyramid,
                                   bool draw_on_level0) const {
  for(std::size_t l=0;l<image_pyramid.size();++l) {
    cv::Mat canvas;
    cv::cvtColor(image_pyramid[l],canvas,cv::COLOR_GRAY2RGB);
    // draw circles for the identified keypoints
    for(std::size_t i=0;i<keypoints_.size();++i) {
      if(!draw_on_level0 && keypoints_[i].level_ != l) {
        continue;
      }
      double scale = draw_on_level0?1.0:(double)(1<<l);
      double x = keypoints_[i].x_ * 1024 / scale;
      double y = keypoints_[i].y_ * 1024 / scale;
      cv::circle(canvas,
                 cv::Point((int)x,(int)y),
                 (keypoints_[i].level_+1)*3*1024,
                 cv::Scalar(0,0,255), // B,G,R
                 1,
                 8,
                 10);
    }
    cv::imshow(title, canvas);
    cv::waitKey();

    if(draw_on_level0) {
      break;
    }
  }
}

} // namespace vilib
