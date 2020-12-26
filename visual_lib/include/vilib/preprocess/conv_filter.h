/*
 * Convolution filtering
 * conv_filter.h
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

#include <iostream>
#include <cuda_runtime_api.h>

namespace vilib {

enum class conv_filter_type {
  // 3x3 filters
  MEAN_FILTER_3X3=0,
  GAUSSIAN_FILTER_3X3,
  PREWITT_GX_FILTER_3X3,
  PREWITT_GY_FILTER_3X3,
  SOBEL_GX_FILTER_3X3,
  SOBEL_GY_FILTER_3X3,
  // 1X3 filters
  MEAN_FILTER_1X3,
  DIFFERENCE_FILTER_1X3,
  GAUSSIAN_FILTER_1X3,
  PREWITT_FILTER_1X3,
  SOBEL_FILTER_1X3
};
typedef enum conv_filter_type conv_filter_type_t;

enum class conv_filter_border_type {
  // skip the elements on the image's border, where we don't have enough information
  BORDER_SKIP=0,
  // 000000|abcdefgh|0000000
  BORDER_ZERO,
  // aaaaaa|abcdefgh|hhhhhhh
  BORDER_REPLICATE,
  // fedcba|abcdefgh|hgfedcb
  BORDER_REFLECT,
  // cdefgh|abcdefgh|abcdefg
  BORDER_WRAP,
  // gfedcb|abcdefgh|gfedcba
  BORDER_REFLECT_101
};
typedef enum conv_filter_border_type conv_filter_border_type_t;

struct filter1x3 {
  float d[3];
};
typedef struct filter3x3 filter3x3_t;

struct filter3x3 {
  float d[9];
};
typedef struct filter1x3 filter1x3_t;

/**
 * Acquire the filter coefficients for the specified 3x3 filter
 * @param filter_type one of the available filter types
 */
const filter3x3_t & conv_filter_get3x3(const conv_filter_type_t filter_type) noexcept;

/**
 * Acquire the filter coefficients for the specified 1x3 filter
 * @param filter_type one of the available filter types
 */
const filter1x3_t & conv_filter_get1x3(const conv_filter_type_t filter_type) noexcept;

/**
 * Apply a separated convolutional filter on a single channel image on the GPU
 * @param d_image_in the input image array pointer
 * @param input_pitch the number of elements per row in the input image
 *        Note: not bytes, elements
 * @param d_image_out the output image array pointer
 * @param output_pitch the number of elements per row in the output image
 *        Note: not bytes, elements
 * @param width_px the width of the image in pixels
 * @param height_px the height of the iamge in pixels
 * @param filter_type_row one of the available 1x3 filter types
 * @param filter_type_col one of the available 1x3 filter types
 * @param border_type one of the available border types
 * @param scale every input is multiplied by this parameter
 * @param stream CUDA stream to be used
 */
template <typename I, typename T, typename O>
void conv_filter_sep_gpu(const I * d_image_in,
                         const int input_pitch,
                         T * d_temp_inout,
                         const int temp_pitch,
                         O * d_image_out,
                         const int output_pitch,
                         const int width_px,
                         const int height_px,
                         const conv_filter_type_t filter_type_row,
                         const conv_filter_type_t filter_type_col,
                         const conv_filter_border_type_t border_type,
                         const float scale = 1.0f,
                         cudaStream_t stream=0);

/**
 * Apply a convolutional filter on a single channel image on the CPU (serial)
 * @param h_image_in the input image array pointer
 * @param input_pitch the number of elements per row in the input image
 *        Note: not bytes, elements
 * @param h_image_out the output image array pointer
 * @param output_pitch the number of elements per row in the output image
 *        Note: not bytes, elements
 * @param width_px the width of the image in pixels
 * @param height_px the height of the iamge in pixels
 * @param filter_type one of the available 3x3 filter types
 * @param border_type one of the available border types
 * @param scale every input element is multiplied by this parameter
 */
void conv_filter_cpu(const unsigned char * h_image_in,
                     const int input_pitch,
                     unsigned char * h_image_out,
                     const int output_pitch,
                     const int width_px,
                     const int height_px,
                     const conv_filter_type_t filter_type,
                     const conv_filter_border_type_t border_type,
                     const float scale = 1.0f);

std::ostream & operator<<(std::ostream & os, const vilib::conv_filter_border_type_t & border_type);

} // namespace vilib