/*
 * Base class for feature detectors
 * detector_base.h
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

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include "vilib/common/occupancy_grid_2d.h"
#include "vilib/common/frame.h"

namespace vilib {

template<bool use_grid>
class DetectorBase {
public:
  // identified interest points
  struct FeaturePoint {
    double x_;
    double y_;
    double score_;
    unsigned int level_;

    FeaturePoint(double x, double y, double score, int level) :
      x_(x),y_(y),score_(score),level_(level) {}
    ~FeaturePoint(void) = default;
  };

  DetectorBase(const std::size_t image_width,
               const std::size_t image_height,
               // width of a cell in pixels (unused if use_grid false)
               const std::size_t cell_size_width,
               // height of a cell in pixels (unused if use_grid false)
               const std::size_t cell_size_height,
               // minimum pyramid level (greatest resolution) where the detection takes place (inclusive)
               const std::size_t min_level,
               // maximum pyramid level (lowest resolution) where the detection takes place (exclusive)
               const std::size_t max_level,
               // number of pixels on the left and right borders to be skipped
               const std::size_t horizontal_border,
               // number of pixels on the top and bottom borders to be skipped
               const std::size_t vertical_border);
  virtual ~DetectorBase(void) = default;

  inline OccupancyGrid2D & getGrid(void) { return grid_; }
  inline const std::vector<struct FeaturePoint> & getPoints(void) const { return keypoints_; }
  inline const std::size_t & getCellSizeWidth(void) { return cell_size_width_; }
  inline const std::size_t & getCellSizeHeight(void) { return cell_size_height_; }
  inline const std::size_t & getCellCountHorizontal(void) { return n_cols_; }
  inline const std::size_t & getCellCountVertical(void) { return n_rows_; }

  virtual void reset(void);
  virtual void detect(const std::vector<cv::Mat> & image);
  void addFeaturePoint(double x, double y, double score, unsigned int level);
  virtual std::size_t count(void) const;

  void displayFeatures(const char * title,
                          const std::vector<cv::Mat> & image_pyramid,
                          bool draw_on_level0 = true,
                          bool draw_cells = true) const;
protected:
  // cell size (width & height)
  std::size_t cell_size_width_;
  std::size_t cell_size_height_;
  // number of cells in the horizontal direction
  std::size_t n_cols_;
  // number of rows in the vertical direction
  std::size_t n_rows_;
  // minimum pyramid level to execute the feature detection on
  std::size_t min_level_;
  // maximum pyramid level TO execute the feature detection on
  std::size_t max_level_;
  // borders (applicable on both sides)
  std::size_t horizontal_border_;
  std::size_t vertical_border_;
  // identified keypoints (n_cols_ x n_rows_)
  std::vector<struct FeaturePoint> keypoints_;
  // occupancy grid
  OccupancyGrid2D grid_;
};

} // namespace vilib
