/*
 * Functions for handling ROS types
 * ros.cpp
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
