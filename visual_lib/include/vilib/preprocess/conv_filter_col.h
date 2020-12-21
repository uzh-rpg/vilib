/*
 * Column-wise convolution filtering
 * conv_filter_col.h
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

#include "vilib/preprocess/conv_filter.h"

namespace vilib {

/**
 * Apply a columm-wise convolutional filter on a single channel image on the GPU
 * @param d_image_in the input image array pointer
 * @param input_pitch the number of elements per row in the input image
 *        Note: not bytes, elements
 * @param d_image_out the output image array pointer
 * @param output_pitch the number of elements per row in the output image
 *        Note: not bytes, elements
 * @param width_px the width of the image in pixels
 * @param height_px the height of the iamge in pixels
 * @param filter_type one of the available 1x3 filter types
 * @param border_type one of the available border types (~OpenCV)
 * @param skip_first_and_last_row skip the first and the last row during convolution
 *                                these lines will be untouched
 * @param scale every input element is multiplied by this parameter
 * @param stream CUDA stream to be used
 */
template <typename I, typename O>
void conv_filter_col_gpu(const I * d_image_in,
                         const int input_pitch,
                         O * d_image_out,
                         const int output_pitch,
                         const int width_px,
                         const int height_px,
                         const conv_filter_type_t filter_type,
                         const conv_filter_border_type_t border_type,
                         const bool skip_first_and_last_row,
                         const float scale = 1.0f,
                         cudaStream_t stream = 0);

/**
 * Apply a column-wise convolutional filter on a single channel image on the CPU (serial)
 * @param h_image_in the input image array pointer
 * @param input_pitch the number of elements per row in the input image
 * @param h_image_out the output image array pointer
 * @param output_pitch the number of elements per row in the output image
 * @param width_px the width of the image in pixels
 * @param height_px the height of the iamge in pixels
 * @param filter_type one of the available 1x3 filter types
 * @param border_type one of the available border types (~OpenCV)
 * @param skip_first_and_last_col skip the first and last columns during convolution
 *                                these columns will be untouched
 * @param scale every input element is multiplied by this parameter
 */
void conv_filter_col_cpu(const unsigned char * h_image_in,
                         const int input_pitch,
                         unsigned char * h_image_out,
                         const int output_pitch,
                         const int width_px,
                         const int height_px,
                         const conv_filter_type_t filter_type,
                         const conv_filter_border_type_t border_type,
                         const bool skip_first_and_last_col,
                         const float scale = 1.0f);

} // namespace vilib