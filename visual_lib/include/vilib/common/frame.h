/*
 * Wrapper class for camera frames
 * frame.h
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
