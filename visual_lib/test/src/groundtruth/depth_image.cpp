/*
 * Depth image
 * depth_image.cpp 
 */

#include <iostream>
#include <unistd.h>
#include "syoyo/tinyexr.h"
#include "test/groundtruth/depth_image.h"
#include "test/arguments.h"

#define DEPTH_INFINITY_THRESHOLD          100000.0f

DepthImage::DepthImage(const char * depth_image_path,
                       const Eigen::VectorXd & K_intr,
                       const Eigen::Matrix4d & T_W_C) :
  valid_(false),T_W_C_(T_W_C) {
  // modify path if it is not an absolute path
  if(depth_image_path[0] != '/') {
    // relative path given
    path_ = get_executable_folder_path();
    path_ += '/';
    path_ += depth_image_path;
  } else {
    // absolute path given
    path_ = depth_image_path;
  }
  // load EXR image
  int ret = LoadEXR(&out_, &width_, &height_, path_.c_str(), &err_msg_);
  if(ret != TINYEXR_SUCCESS) {
    if(err_msg_) {
      std::cout << err_msg_ << std::endl;
      FreeEXRErrorMessage(err_msg_);
    }
  } else {
    valid_ = true;
    max_id_ = width_ * height_ * 4 -1;
    // K_intr [ fx, fy, cx, cy ]
    Eigen::Matrix3d K;
    K << K_intr(0), 0.0, K_intr(2),
         0.0, K_intr(1), K_intr(3),
         0.0, 0.0, 1.0;
    K_inv_ = K.inverse();
  }
}

DepthImage::~DepthImage(void) {
  if(valid_) {
    free(out_);
  }
}

float DepthImage::depth_at(int id) const {
  if(valid_ == false || id > max_id_ || out_[id] > DEPTH_INFINITY_THRESHOLD) {
    return -1.0f;
  }
  return out_[id];
}

float DepthImage::depth_at(int u, int v) const {
  if(valid_ == false || u >= width_ || v >= height_) {
    return -1.0f;
  }
  return this->depth_at((v*width_ + u)*4);
}

float DepthImage::depth_at(float u, float v) const {
  if(valid_ == false || u >= width_ || v >= height_) {
    return -1.0f;
  }
  //maybe do bilinear interpolation later
  //for now, just use the integer part of the coordinates
  return this->depth_at(((int)v*width_ + (int)u)*4);
}

Eigen::Vector4d DepthImage::point_at(int x, int y) const {
  return this->point_at((float)x,(float)y);
}

Eigen::Vector4d DepthImage::point_at(float u, float v) const {
  float z = this->depth_at(u,v);
  Eigen::Vector3d p(u,v,1.0);
  Eigen::Vector3d f = (K_inv_*p);
  f.normalize();
  Eigen::Vector4d p_C(f(0)*z,f(1)*z,f(2)*z,1.0);
  return (T_W_C_*p_C);
}

int DepthImage::width(void) const {
  return width_;
}

int DepthImage::height(void) const {
  return height_;
}
