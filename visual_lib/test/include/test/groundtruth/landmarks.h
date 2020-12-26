/*
 * Landmark functionalities
 * landmarks.h
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

#include <functional>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include "test/groundtruth/blender.h"
#include "test/groundtruth/depth_image.h"

class Landmarks {
public:
  Landmarks(double width_x, double depth_y, double height_z);
  Landmarks(const Blender & blender, std::size_t frame_id);
  Landmarks(const char * depth_image_path,
            const Eigen::VectorXd & K_intr,
            const Eigen::Matrix4d & T_W_C);
  ~Landmarks(void);
  void populate_grid(unsigned int rows, unsigned int cols,
                     bool rand_x=false, bool rand_y=false, bool rand_z=false,
                     double sigma2_x=0.0, double sigma2_y=0.0, double sigma2_z=0.0);
  void keep_only(unsigned int num);
  void visualize(const Eigen::Matrix4d & T_C_W,
                 const Eigen::VectorXd & cam_intr,
                 cv::Mat & image);
  void list(void) const;
  void clear(void);
  std::size_t count(void) const;
  void iterate(std::function<void(const Eigen::Vector4d &)> fn) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  enum class Type : unsigned char {
    VIRTUAL_CUBOID,
    DEPTH_IMAGE
  };

  Type type_;
  double width_x_;
  double depth_y_;
  double height_z_;
  const DepthImage * depth_image_;
  std::size_t landmark_num_;
  Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::ColMajor> landmarks_;
};
