/*
 * Class for holding a particular representation of an input image
 * subframe.h
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

#include <list>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime_api.h>
#ifdef ROS_SUPPORT
#include <sensor_msgs/Image.h>
#endif /* ROS_SUPPORT */

namespace vilib {

class Subframe {
public:
  enum class MemoryType : unsigned char {
    PAGED_HOST_MEMORY=0,
    PINNED_HOST_MEMORY,
    UNIFIED_MEMORY,
    LINEAR_DEVICE_MEMORY,
    PITCHED_DEVICE_MEMORY
  };

  Subframe(std::size_t width,
           std::size_t height,
           std::size_t data_bytes,
           MemoryType type);
  ~Subframe(void);
  void copy_from(const cv::Mat & h_img,
                 bool async = false,
                 cudaStream_t stream_num = 0);
#ifdef ROS_SUPPORT
  void copy_from(const sensor_msgs::ImageConstPtr & h_img,
                 bool async = false,
                 cudaStream_t stream_num = 0);
#endif /* ROS_SUPPORT */
  void copy_to(cv::Mat & h_img,
               bool async = false,
               cudaStream_t stream_num = 0) const;
  void display(void) const;

  std::size_t width_;       // width of a subframe in pixel units
  std::size_t height_;      // height of a subframe in pixel units
  std::size_t data_bytes_;  // representation length of one pixel in byte units
  MemoryType type_;    // underlying memory type
  std::size_t total_bytes_; // total used bytes
  std::size_t pitch_;       // length of a row in byte units
  unsigned char * data_;    // pointer of the buffer
};

} // namespace vilib
