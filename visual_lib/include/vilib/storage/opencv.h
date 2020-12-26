/*
 * Functions for handling OpenCV types
 * opencv.h
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
