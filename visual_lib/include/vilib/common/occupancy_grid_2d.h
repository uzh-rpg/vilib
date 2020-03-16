/*
 * Feature 2D occupancy grid
 * occupancy_grid.h
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
  OccupancyGrid2D(int cell_size, int n_cols, int n_rows)
    : cell_size_(cell_size)
    , n_cols_(n_cols)
    , n_rows_(n_rows)
    , occupancy_(n_cols*n_rows, false) {}
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

  inline void copyOccupancyFrom(const OccupancyGrid2D & other) {
    assert(occupancy_.size() == other.occupancy_.size());
    occupancy_ = other.occupancy_;
  }

  template<typename Derived>
  std::size_t getCellIndex(const Eigen::MatrixBase<Derived>& px) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    return static_cast<std::size_t>((px(1))/cell_size_*n_cols_
                             + (px(0))/cell_size_);
  }

  inline std::size_t getCellIndex(int x, int y, int scale = 1) const {
    return static_cast<std::size_t>((scale*y)/cell_size_*n_cols_
                             + (scale*x)/cell_size_);
  }

  inline void fillWithKeypoints(const Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::ColMajor> & keypoints,
                                const std::size_t & feature_num) {
    /*
     * Note to future self and others:
     * we also use feature_num not just .cols(), because the number of columns
     * can be larger than the actual feature count due to overallocation for
     * allocation efficiency.
     */
    // TODO(cfo): could be implemented using block operations.
    for(std::size_t i = 0; i < feature_num; ++i) {
      occupancy_.at(getCellIndex(static_cast<int>(keypoints(0,i)),
                                 static_cast<int>(keypoints(1,i)), 1)) = true;
    }
  }

  const int cell_size_;
  const int n_cols_;
  const int n_rows_;
  std::vector<bool> occupancy_;
};

} // namespace vilib
