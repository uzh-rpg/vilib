/*
 * Wrapper class for camera frames
 * frame.h
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

#include <vector>
#include <mutex>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#ifdef ROS_SUPPORT
#include <sensor_msgs/Image.h>
#endif /* ROS_SUPPORT */
#include "vilib/common/types.h"
#include "vilib/storage/subframe.h"

namespace vilib {

class Frame {
public:
  Frame(const cv::Mat & img,
        const int64_t timestamp_nsec,
        const std::size_t n_pyr_levels,
        cudaStream_t stream = 0);
#ifdef ROS_SUPPORT
  Frame(const sensor_msgs::ImageConstPtr & msg,
        const std::size_t n_pyr_levels,
        cudaStream_t stream = 0);
#endif /* ROS_SUPPORT */
  ~Frame(void);

  image_pyramid_descriptor_t getPyramidDescriptor(void) const;

  void resizeFeatureStorage(std::size_t new_size);

  // Unique ID of the frame
  std::size_t id_;
  // Timestamp of frame in nanoseconds
  int64_t timestamp_nsec_;
  // Vector holding the image pyramid either in host or GPU memory
  std::vector<std::shared_ptr<Subframe>> pyramid_;

  // Number of successfully extracted features
  std::size_t num_features_ = 0u;
  // Feature coordinates on the image plane in double precision
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::ColMajor> px_vec_;
  // Feature level: the pyramid level where the feature was extracted from
  Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::ColMajor> level_vec_;
  // Features scores: the strongness of the extracted feature
  Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> score_vec_;
  // ID of every observed 3d point. -1 if no point assigned.
  Eigen::VectorXi track_id_vec_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  Frame(const int64_t timestamp_nsec,
        const std::size_t image_width,
        const std::size_t image_height,
        const std::size_t n_pyr_levels);

  static std::size_t getNewId(void);

  // Continuously growing frame counter to provide unique IDs
  static std::size_t last_id_;
  static std::mutex last_id_mutex_;
};

} // namespace vilib
