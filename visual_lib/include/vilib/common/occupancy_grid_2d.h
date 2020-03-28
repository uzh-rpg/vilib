/*
 * Feature 2D occupancy grid
 * occupancy_grid.h
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

// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <assert.h>
#include <algorithm>
#include <Eigen/Dense>
#include <vector>

namespace vilib {

class OccupancyGrid2D {
public:
  OccupancyGrid2D(int cell_size_width,
                  int cell_size_height,
                  int n_cols,
                  int n_rows) :
    cell_size_width_(cell_size_width),
    cell_size_height_(cell_size_height),
    n_cols_(n_cols),
    n_rows_(n_rows),
    occupancy_(n_cols*n_rows, false) {}
  ~OccupancyGrid2D() = default;

  inline void reset(void) {
    std::fill(occupancy_.begin(), occupancy_.end(), false);
  }

  inline std::size_t size(void) const {
    return occupancy_.size();
  }

  inline bool isEmpty(const std::size_t & cell_index) const {
    assert(cell_index < occupancy_.size());
    return (occupancy_[cell_index] == false);
  }

  inline bool isOccupied(const std::size_t & cell_index) const {
    assert(cell_index < occupancy_.size());
    return (occupancy_[cell_index] == true);
  }

  inline void setOccupied(const std::size_t cell_index) {
    assert(cell_index < occupancy_.size());
    occupancy_[cell_index] = true;
  }

  inline void setOccupied(int x, int y, int scale = 1) {
    std::size_t cell_index = getCellIndex(x,y,scale);
    assert(cell_index < occupancy_.size());
    occupancy_[cell_index] = true;
  }

  inline std::size_t getOccupiedCount(void) const {
    return std::count(occupancy_.begin(),occupancy_.end(),true);
  }

  inline std::size_t getCellIndex(int x, int y, int scale = 1) const {
    return static_cast<std::size_t>((scale*y)/cell_size_height_*n_cols_ +
                                    (scale*x)/cell_size_width_);
  }

  const int cell_size_width_;
  const int cell_size_height_;
  const int n_cols_;
  const int n_rows_;
  std::vector<bool> occupancy_;
};

} // namespace vilib
