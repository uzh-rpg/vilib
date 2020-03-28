/*
 * Depth image
 * depth_image.h
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

#include <Eigen/Dense>
#include <string.h>

class DepthImage {
public:
  DepthImage(const char * depth_image_path,
             const Eigen::VectorXd & K_intr,
             const Eigen::Matrix4d & T_W_C);
  ~DepthImage(void);
  float depth_at(int u, int v) const;
  float depth_at(float u, float v) const;
  Eigen::Vector4d point_at(int u, int v) const;
  Eigen::Vector4d point_at(float u, float v) const;
  int width(void) const;
  int height(void) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  float depth_at(int id) const;

  std::string path_;
  int width_;
  int height_;
  int max_id_;
  float * out_; // width x height x RGBA (row-major)
  const char * err_msg_;
  bool valid_;
  Eigen::Matrix3d K_inv_;
  Eigen::Matrix4d T_W_C_;
};
