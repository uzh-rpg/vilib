/*
 * Functions for handling OpenCV types
 * opencv.h
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

#pragma once

#include <opencv2/core/mat.hpp>
#include <cuda_runtime_api.h>

namespace vilib {

/*
 * Copy an OpenCV image/2D matrix to GPU memory buffer
 * @param h_src_img an OpenCV matrix on the CPU
 * @param d_dst_img a memory buffer on the GPU
 * @param dst_img_pitch the memory buffer row-pitch on the GPU in byte units
 * @param async is the copying operation asynchronous?
 * @param stream_num if the copying operation is asynchronous, which GPU stream to use
 */
void opencv_copy_from_image_to_gpu(const cv::Mat & h_src_img,
                                   unsigned char * d_dst_img,
                                   unsigned int dst_img_pitch,
                                   bool async = false,
                                   cudaStream_t stream_num = 0);

/*
 * Copy an OpenCV image/2D matrix to a host memory buffer
 * @param h_src_img an OpenCV matrix on the CPU
 * @param h_dst_img a memory buffer on the GPU
 * @param dst_img_pitch the memory buffer row-pitch on the GPU in byte units
 */
void opencv_copy_from_image_to_host(const cv::Mat & h_src_img,
                                    unsigned char * h_dst_img,
                                    unsigned int dst_img_pitch);

/*
 * Copy from a host memory buffer to an OpenCV image/2D matrix
 * @param h_dst_img destination OpenCV image
 * @param h_src_img source buffer
 * @param src_img_width source image width in pixel units
 * @param src_img_height source image height in pixel units
 * @param src_element_size size of each pixel element in byte units
 * @param src_img_pitch the memory buffer row-pitch in byte units
 */
void opencv_copy_from_host_to_image(cv::Mat & h_dst_img,
                                    const unsigned char * h_src_img,
                                    unsigned int src_img_width,
                                    unsigned int src_img_height,
                                    unsigned int src_element_size,
                                    unsigned int src_img_pitch);

/*
 * Copy from a GPU memory buffer to an OpenCV image/2D matrix
 * @param h_dst_img destination OpenCV image
 * @param h_src_img source buffer
 * @param src_img_width source image width in pixel units
 * @param src_img_height source image height in pixel units
 * @param src_element_size size of each pixel element in byte units
 * @param src_img_pitch the memory buffer row-pitch in byte units
 * @param async is the copying operation asynchronous?
 * @param stream_num if the copying operation is asynchronous, which GPU stream to use
 */
void opencv_copy_from_gpu_to_image(cv::Mat & h_dst_img,
                                   const unsigned char * d_src_img,
                                   unsigned int src_img_width,
                                   unsigned int src_img_height,
                                   unsigned int src_element_size,
                                   unsigned int src_img_pitch,
                                   bool async,
                                   cudaStream_t stream_num);

/*
 * Register image in pinned-memory so that the memory transfers become faster
 * @param h_img image to be registered in pinned memory
 */
void opencv_register_image_in_pinned_memory(const cv::Mat & h_img);

/*
 * Unregister image in pinned-memory so that the memory transfers become faster
 * @param h_img image to be registered in pinned memory
 */
void opencv_unregister_image_from_pinned_memory(const cv::Mat & h_img);

} // namespace vilib
