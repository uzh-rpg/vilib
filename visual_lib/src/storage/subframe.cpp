/*
 * Class for holding a particular representation of an input image
 * subframe.cpp
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

#include <stdlib.h>
#include <string>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#ifdef ROS_SUPPORT
#include <sensor_msgs/image_encodings.h>
#endif /* ROS_SUPPORT */
#include "vilib/storage/subframe.h"
#include "vilib/storage/subframe_pool.h"
#include "vilib/storage/opencv.h"
#include "vilib/storage/ros.h"

namespace vilib {

Subframe::Subframe(std::size_t width,
                   std::size_t height,
                   std::size_t data_bytes,
                   MemoryType type) :
  width_(width),height_(height),data_bytes_(data_bytes),type_(type) {
  // perform the memory allocations
  switch(type) {
    case MemoryType::PAGED_HOST_MEMORY: {
      total_bytes_ = width*data_bytes*height; // packed: width * height * data_bytes
      data_ = (unsigned char *)malloc(total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::PINNED_HOST_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMallocHost((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMalloc((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      cudaMallocPitch((void**)&data_,&pitch_,width*data_bytes,height);
      /*
       * Note to future self:
       * the returned pitch will be the calculated pitch in byte units
       */
      total_bytes_ = pitch_*height;
      break;
    }
    case MemoryType::UNIFIED_MEMORY: {
      total_bytes_ = width*data_bytes*height; // packed: width * height * data_bytes
      cudaMallocManaged((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
  }
}

Subframe::~Subframe(void) {
  // perform the memory deallocations
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY: {
      free(data_);
      break;
    }
    case MemoryType::PINNED_HOST_MEMORY: {
      cudaFreeHost(data_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      cudaFree(data_);
      break;
    }
  }
}

void Subframe::copy_from(const cv::Mat & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO : support color!
  assert(h_img.channels() == 1);
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_image_to_host(h_img,
                                     data_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_image_to_gpu(h_img,
                                    data_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

#ifdef ROS_SUPPORT
void Subframe::copy_from(const sensor_msgs::ImageConstPtr & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO: support color
  assert(h_img->encoding == sensor_msgs::image_encodings::MONO8);
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      ros_copy_from_image_to_host(h_img,
                                  1,
                                  data_,
                                  pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      ros_copy_from_image_to_gpu(h_img,
                                 1,
                                 data_,
                                 pitch_,
                                 async,
                                 stream_num);
      break;
    }
  }
}
#endif /* ROS_SUPPORT */

void Subframe::copy_to(cv::Mat & h_img,
                       bool async,
                       cudaStream_t stream_num) const {
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_host_to_image(h_img,
                                     data_,
                                     width_,
                                     height_,
                                     data_bytes_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_gpu_to_image(h_img,
                                    data_,
                                    width_,
                                    height_,
                                    data_bytes_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

void Subframe::display(void) const {
  // copy image to a temporary buffer and display that
  std::string subframe_title("Subframe (");
  subframe_title += std::to_string(width_);
  subframe_title += "x";
  subframe_title += std::to_string(height_);
  subframe_title += ")";
  cv::Mat image;
  copy_to(image);
  cv::imshow(subframe_title.c_str(), image);
  cv::waitKey();
}

} // namespace vilib
