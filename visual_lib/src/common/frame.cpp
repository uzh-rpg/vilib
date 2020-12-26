/*
 * Wrapper class for camera frames
 * frame.cpp
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

#ifdef ROS_SUPPORT
#include <sensor_msgs/image_encodings.h>
#endif /* ROS_SUPPORT */
#include "vilib/common/frame.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/preprocess/image_preprocessing.h"
#include "vilib/preprocess/pyramid.h"
#include "vilib/config.h"

namespace vilib {

std::size_t Frame::last_id_ = 0;
std::mutex Frame::last_id_mutex_;

Frame::Frame(const cv::Mat & img,
             const int64_t timestamp_nsec,
             const std::size_t n_pyr_levels,
             cudaStream_t stream) :
  Frame(timestamp_nsec,
        img.cols,
        img.rows,
        n_pyr_levels) {
  preprocess_image(img,pyramid_,stream);
}

#ifdef ROS_SUPPORT
Frame::Frame(const sensor_msgs::ImageConstPtr & msg,
             const std::size_t n_pyr_levels,
             cudaStream_t stream) :
  Frame(msg->header.stamp.sec*1e9 + msg->header.stamp.nsec,
        msg->width,
        msg->height,
        n_pyr_levels) {
  preprocess_image(msg,pyramid_,stream);
}
#endif /* ROS_SUPPORT */

Frame::Frame(const int64_t timestamp_nsec,
             const std::size_t image_width,
             const std::size_t image_height,
             const std::size_t n_pyr_levels) :
  id_(getNewId()),
  timestamp_nsec_(timestamp_nsec) {
  /*
   * Note: we allocate space for a grayscale image,
   *       irrespective of the input image
   */
  PyramidPool::get(IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM,
                   image_width,
                   image_height,
                   1,
                   n_pyr_levels,
                   IMAGE_PYRAMID_MEMORY_TYPE,
                   pyramid_);
}

std::size_t Frame::getNewId(void) {
  std::lock_guard<std::mutex> lock(last_id_mutex_);
  return last_id_++;
}

Frame::~Frame(void) {
  // return the pyramid buffers
  PyramidPool::release(pyramid_);
}

image_pyramid_descriptor_t Frame::getPyramidDescriptor(void) const {
  image_pyramid_descriptor_t i;
  i.desc = PyramidPool::get_descriptor();
  for(std::size_t l=0;l<pyramid_.size();++l) {
    i.data[l] = pyramid_[l]->data_;
  }
  return i;
}

void Frame::resizeFeatureStorage(std::size_t new_size) {
  // Note: we dont want to lose features during this function call
  assert((int)new_size >= px_vec_.cols());
  // Don't do anything if the column size is the same
  if((int)new_size == px_vec_.cols()) {
    return;
  }
  std::size_t uninitialized_cols = new_size - num_features_;
  // do the resizing
  px_vec_.conservativeResize(Eigen::NoChange, new_size);
  level_vec_.conservativeResize(new_size, Eigen::NoChange);
  level_vec_.tail(uninitialized_cols).setZero();
  score_vec_.conservativeResize(new_size, Eigen::NoChange);
  track_id_vec_.conservativeResize(new_size);
  track_id_vec_.tail(uninitialized_cols).setConstant(-1);
}

} // namespace vilib
