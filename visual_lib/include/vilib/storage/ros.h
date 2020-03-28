/*
 * Functions for handling ROS types
 * ros.h
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
#pragma once

#include <sensor_msgs/Image.h>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime_api.h>

namespace vilib {

/*
 * Copy a ROS sensor image to an OpenCV matrix
 * @param h_src_img the image to copy from
 * @param h_dst_img the OpenCV matrix to copy to
 */
void ros_to_opencv_mat(const sensor_msgs::ImageConstPtr & h_src_img,
                       cv::Mat & h_dst_img);

/*
 * Copy a ROS sensor image to the GPU
 * @param h_src_img the image to copy from
 * @param src_img_element_size size of each pixel element in byte units
 * @param d_dst_img the GPU memory buffer pointer
 * @param dst_img_pitch the allocated GPU memory row length in byte units
 * @param async is the copying operation asynchronous?
 * @param stream_num if the copying operation is asynchronous, which GPU stream to use
 */
void ros_copy_from_image_to_gpu(const sensor_msgs::ImageConstPtr & h_src_img,
                                uint8_t src_img_element_size,
                                unsigned char * d_dst_img,
                                unsigned int dst_img_pitch,
                                bool async = false,
                                cudaStream_t stream_num = 0);

/*
 * Copy a ROS sensor image to a host memory buffer
 * @param h_src_img the image to copy from
 * @param src_img_element_size size of each pixel element in byte units
 * @param h_dst_img the GPU memory pointer
 * @param dst_img_pitch the allocated GPU memory row length in byte units
 */
void ros_copy_from_image_to_host(const sensor_msgs::ImageConstPtr & h_src_img,
                                 uint8_t src_img_element_size,
                                 unsigned char * h_dst_img,
                                 unsigned int dst_img_pitch);

} // namespace vilib

#endif /* ROS_SUPPORT */
