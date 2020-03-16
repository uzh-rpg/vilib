/*
 * Depth image
 * depth_image.h
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
