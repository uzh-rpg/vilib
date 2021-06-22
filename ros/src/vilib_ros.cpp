/*
 * Copyright (c) 2019-2021 Philipp Foehn,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "vilib/vilib_ros.hpp"

#include <ros/ros.h>

#include <unordered_map>

#include "vilib/common/frame.h"
#include "vilib/common/framebundle.h"
#include "vilib/storage/pyramid_pool.h"


namespace vilib {


VilibRos::VilibRos(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
  : params_(pnh), nh_(nh), pnh_(pnh), it_(pnh_) {
  if (!params_.valid()) {
    ROS_ERROR("Could not load valid parameter set!");
    ros::shutdown();
  }

  timer_frame_.nest(timer_copy_);
  timer_frame_.nest(timer_tracking_);
  timer_frame_.nest(timer_visualization_);

  image_sub_ = it_.subscribe("image", 1, &VilibRos::imageCallback, this);
  image_pub_ = it_.advertise("debug_image", 1, false);

  PyramidPool::init(1, params_.image_width, params_.image_height, 1,
                    params_.numberOfPyramidLevels(), IMAGE_PYRAMID_MEMORY_TYPE);

  feature_tracker_ = params_.createFeatureTracker();
  feature_detector_ = params_.createFeatureDetector();
  feature_tracker_->setDetectorGPU(feature_detector_, 0);

  process_thread_ = std::thread(&VilibRos::processThread, this);
}

VilibRos::~VilibRos() {
  if (process_thread_.joinable()) {
    shutdown_ = true;
    process_thread_.join();
  }

  PyramidPool::deinit();
}

void VilibRos::imageCallback(const sensor_msgs::ImageConstPtr &msg) {
  const cv_bridge::CvImagePtr image = cv_bridge::toCvCopy(msg, "mono8");
  {
    std::lock_guard<std::mutex> guard(image_queue_mtx_);
    image_queue_.push(image->image);
  }
  image_cv_.notify_all();
}

void VilibRos::processThread() {
  std::mutex mtx;
  cv_bridge::CvImage debug_image;
  debug_image.image =
    cv::Mat(params_.image_height, params_.image_width, CV_8UC3);
  debug_image.encoding = "bgr8";
  while (ros::ok() || !shutdown_) {
    std::unique_lock<std::mutex> lock(mtx);
    image_cv_.wait_for(lock, std::chrono::milliseconds(100));

    while (true) {
      timer_frame_.tic();
      static std::shared_ptr<Frame> frame;
      {
        std::lock_guard<std::mutex> guard(image_queue_mtx_);
        if (image_queue_.empty()) break;
        timer_copy_.tic();
        frame = std::make_shared<Frame>(image_queue_.front(), 0,
                                        params_.numberOfPyramidLevels());

        if (params_.publish_debug_image)
          cv::cvtColor(image_queue_.front(), debug_image.image, CV_GRAY2BGR);
        timer_copy_.toc();
        image_queue_.pop();
      }

      std::shared_ptr<FrameBundle> frame_bundle = std::make_shared<FrameBundle>(
        std::vector<std::shared_ptr<Frame>>({frame}));

      timer_tracking_.tic();
      size_t n_tracked, n_detected;
      feature_tracker_->track(frame_bundle, n_tracked, n_detected);
      timer_tracking_.toc();

      if (params_.publish_debug_image) {
        timer_visualize_.tic();
        debug_image.header.stamp = ros::Time::now();
        visualize(frame, debug_image.image);
        image_pub_.publish(debug_image.toImageMsg());
        timer_visualize_.toc();
      }
      timer_frame_.toc();
    }
  }
}

void VilibRos::visualize(const std::shared_ptr<Frame> &frame,
                         cv::Mat &image) const {
  static int last_track_id = -1;
  static std::unordered_map<int, cv::Scalar> track_colors;

  const int cw = params_.cell_width;
  const int ch = params_.cell_height;
  const int n_rows = (image.rows + ch - 1) / ch;
  const int n_cols = (image.cols + cw - 1) / cw;
  for (int r = 0; r < n_rows; ++r) {
    for (int c = 0; c < n_cols; ++c) {
      cv::rectangle(image, cv::Point(c * cw, r * ch),
                    cv::Point((c + 1) * cw, (r + 1) * ch),
                    cv::Scalar(244, 215, 66),  // B,G,R
                    1, 8, 0);
    }
  }

  // note: id-s start from 0
  for (int i = 0; i < frame->num_features_; ++i) {
    const Eigen::Matrix<int, 1, 2> pos = frame->px_vec_.col(i).cast<int>();

    const int track_id = frame->track_id_vec_[i];

    cv::Scalar track_color(255, 255, 255);
    if (last_track_id < track_id) {
      track_colors[track_id] =
        cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    } else {
      track_color = track_colors[track_id];
    }

    cv::circle(image, cv::Point(pos.x(), pos.y()), 3, track_color, 3, 8);
  }
  // update the highest track id
  if (frame->num_features_ > 0 &&
      frame->track_id_vec_[frame->num_features_ - 1] > last_track_id) {
    last_track_id = frame->track_id_vec_[frame->num_features_ - 1];
  }
}

}  // namespace vilib

int main(int argc, char **argv) {
  ros::init(argc, argv, "vilib");
  vilib::VilibRos vilib;

  ros::spin();
  return 0;
}
