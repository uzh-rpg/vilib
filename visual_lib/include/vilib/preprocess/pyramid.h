/*
 * Functions for creating image pyramids
 * pyramid.h
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

#include <vector>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime_api.h>
#include "vilib/storage/subframe.h"

namespace vilib {

/*
 * Create an image pyramid on the CPU
 * The images composing the pyramid will have a decreasing resolution
 * (by always halving the input image's resolution). The Input image will be
 * copied to the zeroth level
 * @param h_image_in the original image (level 0)
 * @param levels number of pyramid levels (at least 1)
 * @param h_image_pyramid output image pyramid vector, containing the original
 *                        and the downsampled images
 * @param deep_copy If h_image_in needs to be copied deeply to the zeroth level,
 *                  set this argument to true. Otherwise the copy is only a shallow
 *                  copy.
 */
void pyramid_create_cpu(const cv::Mat & h_image_in,
                        std::vector<cv::Mat> & h_image_pyramid,
                        unsigned int levels,
                        bool deep_copy);

/*
 * Create an image pyramid on the CPU
 * The images composing the pyramid will have a decreasing resolution
 * (by always halving the input image's resolution). The input image should be
 * present on the zeroth level, and the pyramid should already have the right size.
 * @param levels number of pyramid levels (at least 1)
 * @param h_image_pyramid output image pyramid vector, containing the original
 *                        and the downsampled images
 */
void pyramid_create_cpu(std::vector<cv::Mat> & h_image_pyramid);

/*
 * Create an image pyramid on the GPU using vectors
 * The images composing the pyramid will have a decreasing resolution
 * (by always halving the input image's resolution)
 * @param d_images the preallocated image vector holding only the level0 image
 * @param width the precalculated width vector containing the image widths
 * @param height the precalculated height vector containing the image heights
 * @param pitch the precalculated pitch vector containing the image storage pitches
 * @param levels the number of image pyramid levels
 */
void pyramid_create_gpu(std::vector<unsigned char *> & d_images,
                        std::vector<std::size_t> & width,
                        std::vector<std::size_t> & height,
                        std::vector<std::size_t> & pitch,
                        unsigned int levels,
                        cudaStream_t stream);

/*
 * Create an image pyramid on the GPU using a frame wrapper
 * The images composing the pyramid will have a decreasing resolution
 * (by always halving the input image's resolution)
 * It is expected that the subframe buffers lie in the GPU memory
 * @param d_subframes a vector of preallocated Subframes (level 0 is deemed the original image)
 */
void pyramid_create_gpu(std::vector<std::shared_ptr<Subframe>> & d_subframes,
                        cudaStream_t stream);

/*
 * Display an image pyramid that resides in a vector of OpenCV matrices
 * @param pyramid a vector of preallocated OpenCV matrices
 */
void pyramid_display(const std::vector<cv::Mat> & pyramid);

/*
 * Display an image pyramid that resides in a vector of subframes
 * There's no restriction on the memory location of the subframes.
 * @param subframes a vector of preallocated Subframes
 */
void pyramid_display(const std::vector<std::shared_ptr<Subframe>> & subframes);

} // namespace vilib
