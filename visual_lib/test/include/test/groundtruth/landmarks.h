/*
 * Landmark functionalities
 * landmarks.h
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
