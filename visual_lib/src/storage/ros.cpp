/*
 * Functions for handling ROS types
 * ros.cpp
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
#include "vilib/storage/ros.h"
#include "vilib/cuda_common.h"

namespace vilib {

static inline void ros_copy_from_image_common(const sensor_msgs::ImageConstPtr & src_img,
                                              uint8_t src_img_element_size,
                                              unsigned char * dst_img,
                                              unsigned int dst_img_pitch,
                                              bool async,
                                              cudaStream_t stream_num,
                                              enum cudaMemcpyKind dir) {
  void * dst = (void*)dst_img;
  const void * src = (const void *)src_img->data.data();
  std::size_t dpitch = dst_img_pitch;
  std::size_t spitch = src_img->step;
  std::size_t width  = src_img->width * src_img_element_size;
  std::size_t height = src_img->height;
  if(async) {
          CUDA_API_CALL(cudaMemcpy2DAsync(dst,dpitch,
                                          src,spitch,
                                          width,height,
                                          dir,stream_num));
  } else {
          CUDA_API_CALL(cudaMemcpy2D(dst,dpitch,
                                     src,spitch,
                                     width,height,
                                     dir));
  }
}

void ros_to_opencv_mat(const sensor_msgs::ImageConstPtr & h_src_img,
                       cv::Mat & h_dst_img) {
  if(h_src_img->encoding == sensor_msgs::image_encodings::MONO8) {
    h_dst_img = cv::Mat(h_src_img->height,h_src_img->width,CV_8UC1);
    memcpy((void*)h_dst_img.data,
           (const void*)h_src_img->data.data(),
           h_src_img->width * h_src_img->height);
  } else {
    assert(0 && "This encoding is not supported at the moment");
  }
}

void ros_copy_from_image_to_gpu(const sensor_msgs::ImageConstPtr & h_src_img,
                                uint8_t src_img_element_size,
                                unsigned char * d_dst_img,
                                unsigned int dst_img_pitch,
                                bool async,
                                cudaStream_t stream_num) {
  ros_copy_from_image_common(h_src_img,
                             src_img_element_size,
                             d_dst_img,
                             dst_img_pitch,
                             async,
                             stream_num,
                             cudaMemcpyHostToDevice);
}

void ros_copy_from_image_to_host(const sensor_msgs::ImageConstPtr & h_src_img,
                                uint8_t src_img_element_size,
                                unsigned char * h_dst_img,
                                unsigned int dst_img_pitch) {
  ros_copy_from_image_common(h_src_img,
                             src_img_element_size,
                             h_dst_img,
                             dst_img_pitch,
                             false,
                             0,
                             cudaMemcpyHostToHost);
}

} // namespace vilib

#endif /* ROS_SUPPORT */
