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

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib_tracker/logger.hpp"
#include "vilib_tracker/statistic.hpp"
#include "vilib_tracker/timer.hpp"
#include "vilib_tracker/vilib_params.hpp"

namespace vilib {

class VilibRos {
 public:
  VilibRos(const ros::NodeHandle &nh = ros::NodeHandle(),
           const ros::NodeHandle &pnh = ros::NodeHandle("~"));
  ~VilibRos();

  void imageCallback(const sensor_msgs::ImageConstPtr &msg);

 private:
  void processThread();
  void publishFeatures(const ros::Time &frame_time,
                       const std::shared_ptr<Frame> &frame) const;
  void visualize(const std::shared_ptr<Frame> &frame, cv::Mat &image) const;

  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher features_pub_;

  VilibParams params_;

  // Image Queue
  std::queue<cv_bridge::CvImageConstPtr> image_queue_;
  std::mutex image_queue_mtx_;
  std::condition_variable image_cv_;

  // Vilib
  std::shared_ptr<FeatureTrackerGPU> feature_tracker_;
  std::shared_ptr<DetectorBaseGPU> feature_detector_;

  // Thread
  bool shutdown_ = false;
  std::thread process_thread_;

  // Logging and Statistics
  Logger logger_{"Vilib"};
  Statistic<double> stats_tracked_{"Tracked Features"};
  Statistic<double> stats_detected_{"Detected Features"};
  Timer<double> timer_frame_{"Frame Processing"};
  Timer<double> timer_copy_{"Copy"};
  Timer<double> timer_tracking_{"Tracking"};
  Timer<double> timer_visualize_{"Visualize"};
};

}  // namespace vilib
