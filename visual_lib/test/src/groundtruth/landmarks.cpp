/*
 * Landmark functionalities
 * landmarks.cpp
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


#include <random>
#include <sys/time.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "test/groundtruth/landmarks.h"

Landmarks::Landmarks(double width_x, double depth_y, double height_z) :
  type_(Type::VIRTUAL_CUBOID),
  width_x_(width_x),
  depth_y_(depth_y),
  height_z_(height_z),
  landmark_num_(0) {
}

Landmarks::Landmarks(const Blender & blender, std::size_t frame_id) :
  type_(Type::DEPTH_IMAGE),
  landmark_num_(0) {
  const struct Blender::BlenderFrame & bf = blender.getFrame(frame_id);
  const Eigen::Vector4d & camera_intr = blender.getIntrinsicParameters();
  depth_image_ = new DepthImage(bf.image_depth_path_.c_str(),
                                camera_intr,
                                bf.T_W_C_);
}

Landmarks::Landmarks(const char * depth_image_path,
                     const Eigen::VectorXd & K_intr,
                     const Eigen::Matrix4d & T_W_C) :
  type_(Type::DEPTH_IMAGE),
  landmark_num_(0) {
  depth_image_ = new DepthImage(depth_image_path,K_intr,T_W_C);
}

Landmarks::~Landmarks(void) {
  if(type_ == Type::DEPTH_IMAGE) {
    delete depth_image_;
  }
}

void Landmarks::populate_grid(unsigned int rows, unsigned int cols,
                              bool rand_x, bool rand_y, bool rand_z,
                              double sigma2_x, double sigma2_y, double sigma2_z) {
  if(type_ == Type::VIRTUAL_CUBOID) {
    std::mt19937 rd;
    //populating points with respect to the origin of the bounding cuboid
    double width_spacing = width_x_/(cols+1);
    double height_spacing = height_z_/(rows+1);
    double depth_spacing = depth_y_/2.0;
    landmarks_.resize(Eigen::NoChange, rows*cols);
    landmark_num_ = rows*cols;
    for(unsigned int r=0;r<rows;++r) {
      for(unsigned int c=0;c<cols;++c) {
        std::normal_distribution<double> nm_x(0.0,sigma2_x);
        std::normal_distribution<double> nm_y(0.0,sigma2_y);
        std::normal_distribution<double> nm_z(0.0,sigma2_z);
        double coord_x = (c+1)*width_spacing  + (rand_x ? nm_x(rd) : 0.0);
        double coord_y = depth_spacing        + (rand_y ? nm_y(rd) : 0.0);
        double coord_z = (r+1)*height_spacing + (rand_z ? nm_z(rd) : 0.0);
        double coord_w = 1.0;
        landmarks_.col(r*cols + c) = Eigen::Vector4d(coord_x,coord_y,coord_z,coord_w);
      }
    }
  } else if(type_ == Type::DEPTH_IMAGE) {
    //selecting points on the depth map uniformly
    unsigned int depth_image_width = depth_image_->width();
    unsigned int depth_image_height = depth_image_->height();
    double cell_width  = ((double)depth_image_width)  / cols;
    double cell_height = ((double)depth_image_height) / rows;
    landmarks_.resize(Eigen::NoChange, rows*cols);
    landmark_num_ = 0;
    for(unsigned int r=0;r<rows;++r) {
      for(unsigned int c=0;c<cols;++c) {
        //get depth in world frame -> if exists -> take point, otherwise just skip
        int target_row = (int)(r*cell_height + cell_height/2);
        int target_col = (int)(c*cell_width  + cell_width/2);
        float depth = depth_image_->depth_at(target_col,target_row);
        if(depth < 0) {
          continue;
        }
        landmarks_.col(landmark_num_) = depth_image_->point_at(target_col,target_row);
        ++landmark_num_;
      }
    }
  }
}

void Landmarks::keep_only(unsigned int num) {
  if(landmark_num_ > num) {
    landmark_num_= num;
  }
}

void Landmarks::list(void) const {
  for(std::size_t i=0;i<landmark_num_;++i) {
    const Eigen::Vector4d & lm = landmarks_.col(i);
    std::cout << lm(0) << ", " << lm(1) << ", " << lm(2) << ", " << lm(3) << std::endl;
  }
}

void Landmarks::clear(void) {
  landmark_num_=0;
  landmarks_.resize(Eigen::NoChange,0);
}

void Landmarks::iterate(std::function<void(const Eigen::Vector4d &)> fn) const {
  for(std::size_t i=0;i<landmark_num_;++i) {
    // p_W
    fn(landmarks_.col(i));
  }
}

void Landmarks::visualize(const Eigen::Matrix4d & T_C_W,
               const Eigen::VectorXd & cam_intr,
               cv::Mat & image) {
  /*
   * Note to future self:
   * cv::Mat's copy constructor only creates a shallow copy
   */
  cv::Mat canvas = image.clone();
  // create projection matrix M
  Eigen::MatrixXd M(3,4);
  Eigen::MatrixXd P(3,4);
  Eigen::Matrix3d K;
  P << 1,0,0,0,
       0,1,0,0,
       0,0,1,0;
  // cam_intr [ fx, fy, cx, cy ]
  K << cam_intr(0), 0.0, cam_intr(2),
       0.0, cam_intr(1), cam_intr(3),
       0.0, 0.0, 1.0;
  M = K * P * T_C_W;
  const int radius = 1;
  const int margin = 0;
  for(std::size_t i=0;i<landmark_num_;++i) {
    const Eigen::Vector4d & p_W = landmarks_.col(i);
    Eigen::Vector3d p_center = M * p_W;
    float u = p_center(0)/p_center(2);
    float v = p_center(1)/p_center(2);
    if(u >= margin && v >= margin && u < (canvas.cols-margin) && v < (canvas.rows-margin)) {
      cv::circle(canvas,cv::Point(u,v),radius,cv::Scalar(255,0,0),-1,8,0);
    }
  }
  cv::imshow("Visualized landmarks", canvas);
  cv::waitKey();
}

std::size_t Landmarks::count(void) const {
  return landmark_num_;
}
