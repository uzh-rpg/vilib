/*
 * Functions for handling ROS types
 * ros.h
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
