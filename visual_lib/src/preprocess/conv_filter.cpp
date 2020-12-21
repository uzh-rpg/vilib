/*
 * Convolution filtering
 * conv_filter.cu
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

#include <cassert>
#include <algorithm>
#include "vilib/preprocess/conv_filter.h"
#include "vilib/preprocess/conv_filter_col.h"
#include "vilib/preprocess/conv_filter_row.h"

namespace vilib {

#define INSTANTIATE_2D_SEPARABLE(I, T, O)                                                             \
  template __host__ void conv_filter_sep_gpu<I, T, O>(const I * d_image_in,                           \
                                                      const int input_pitch,                          \
                                                      T * d_temp_inout,                               \
                                                      const int temp_pitch,                           \
                                                      O * d_image_out,                                \
                                                      const int output_pitch,                         \
                                                      const int width_px,                             \
                                                      const int height_px,                            \
                                                      const conv_filter_type_t filter_type_row,       \
                                                      const conv_filter_type_t filter_type_col,       \
                                                      const conv_filter_border_type_t border_type,    \
                                                      const float scale,                              \
                                                      cudaStream_t stream)

// 3x3 filters
static const filter3x3_t mean_filter_3x3 { .d = {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                                                 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                                                 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f}};
static const filter3x3_t gaussian_filter_3x3 { .d = {0.0625f, 0.125f, 0.0625f,
                                                     0.1250f, 0.250f, 0.1250f,
                                                     0.0625f, 0.125f, 0.0625f}};
static const filter3x3_t prewitt_gx_filter_3x3 { .d = {-1.0f, 0.0f, +1.0f,
                                                       -1.0f, 0.0f, +1.0f,
                                                       -1.0f, 0.0f, +1.0f}};
static const filter3x3_t prewitt_gy_filter_3x3 { .d = {-1.0f, -1.0f, -1.0f,
                                                       0.0f, 0.0f, 0.0f,
                                                       +1.0f, +1.0f, +1.0f}};
static const filter3x3_t sobel_gx_filter_3x3 { .d = {-1.0f, 0.0f, +1.0f,
                                                     -2.0f, 0.0f, +2.0f,
                                                     -1.0f, 0.0f, +1.0f}};
static const filter3x3_t sobel_gy_filter_3x3 { .d = {-1.0f, -2.0f, -1.0f,
                                                     0.0f, 0.0f, 0.0f,
                                                     +1.0f, +2.0f, +1.0f}};
// 1x3 filters
static const filter1x3_t diff_filter_1x3 { .d = {-1.0f, 0.0f, 1.0f }};
static const filter1x3_t prewitt_filter_1x3 { .d = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f }};
static const filter1x3_t sobel_filter_1x3 { .d = {0.25f, 0.5f, 0.25f }};
static const filter1x3_t gaussian_filter_1x3 { .d = {0.25f, 0.50f, 0.25f }};
static const filter1x3_t mean_filter_1x3 { .d = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f }};

const filter3x3_t & conv_filter_get3x3(const conv_filter_type_t filter_type) noexcept {
  switch(filter_type) {
    case conv_filter_type::MEAN_FILTER_3X3:
      return mean_filter_3x3;
    case conv_filter_type::GAUSSIAN_FILTER_3X3:
      return gaussian_filter_3x3;
    case conv_filter_type::PREWITT_GX_FILTER_3X3:
      return prewitt_gx_filter_3x3;
    case conv_filter_type::PREWITT_GY_FILTER_3X3:
      return prewitt_gy_filter_3x3;
    case conv_filter_type::SOBEL_GX_FILTER_3X3:
      return sobel_gx_filter_3x3;
    case conv_filter_type::SOBEL_GY_FILTER_3X3:
      return sobel_gy_filter_3x3;
    default:
      // nope, it does not exist
      assert(0);
      return mean_filter_3x3;
  }
}
  
const filter1x3_t & conv_filter_get1x3(const conv_filter_type_t filter_type) noexcept {
  switch(filter_type) {
    case conv_filter_type::DIFFERENCE_FILTER_1X3:
      return diff_filter_1x3;
    case conv_filter_type::GAUSSIAN_FILTER_1X3:
      return gaussian_filter_1x3;
    case conv_filter_type::PREWITT_FILTER_1X3:
      return prewitt_filter_1x3;
    case conv_filter_type::SOBEL_FILTER_1X3:
      return sobel_filter_1x3;
    case conv_filter_type::MEAN_FILTER_1X3:
      return mean_filter_1x3;
    default:
      // nope, it does not exist
      assert(0);
      return diff_filter_1x3;
  }
}

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
                         const float scale,
                         cudaStream_t stream) {
  conv_filter_col_gpu<I,T>(d_image_in,
                            input_pitch,
                            d_temp_inout,
                            temp_pitch,
                            width_px,
                            height_px,
                            filter_type_col,
                            border_type,
                            false,
                            1.0f,
                            stream);
  conv_filter_row_gpu<T,O>(d_temp_inout,
                            temp_pitch,
                            d_image_out,
                            output_pitch,
                            width_px,
                            height_px,
                            filter_type_row,
                            border_type,
                            (border_type==conv_filter_border_type::BORDER_SKIP),
                            scale,
                            stream);
}

void conv_filter_cpu(const unsigned char * h_image_in,
                              const int input_pitch,
                              unsigned char * h_image_out,
                              const int output_pitch,
                              const int width_px,
                              const int height_px,
                              const conv_filter_type_t filter_type,
                              const conv_filter_border_type_t border_type,
                              const float scale) {
  const filter3x3_t & filter = conv_filter_get3x3(filter_type);
  int y_min = 0;
  int y_max = height_px-1;
  int x_min = 0;
  int x_max = width_px-1;
  // Note: for border-skip we do not touch the first/last columns
  //       and first/last rows
  if(border_type == conv_filter_border_type::BORDER_SKIP) {
    ++y_min;
    --y_max;
    ++x_min;
    --x_max;
  }
  for(int y=y_min; y<=y_max; ++y) {
    for(int x=x_min; x<=x_max; ++x) {
      float accu = 0.0f;
      for(int f_y=0;f_y<3;++f_y) {
        int i_y = y+f_y-1;
        switch(border_type) {
          case conv_filter_border_type::BORDER_SKIP:
            // nothing to do
            break;
          case conv_filter_border_type::BORDER_ZERO:        // 000000|abcdefgh|0000000
            // nothing to do
            break;
          case conv_filter_border_type::BORDER_REPLICATE:   // aaaaaa|abcdefgh|hhhhhhh
            i_y = std::min(std::max(i_y,0),y_max);
            break;
          case conv_filter_border_type::BORDER_REFLECT:     // fedcba|abcdefgh|hgfedcb
            if(i_y < y_min) {
              i_y = -1*i_y - 1;
            } else if(i_y > y_max) {
              i_y = y_max - (i_y-height_px);
            }
            break;
          case conv_filter_border_type::BORDER_WRAP:        // cdefgh|abcdefgh|abcdefg
            if(i_y < 0) {
              i_y += height_px;
            } else if(i_y > y_max) {
              i_y -= height_px;
            }
            break;
          case conv_filter_border_type::BORDER_REFLECT_101: // gfedcb|abcdefgh|gfedcba
            if(i_y < 0) {
              i_y *= -1;
            } else if(i_y > y_max) {
              i_y = 2*y_max - i_y;
            }
            break;  
        }
        for(int f_x=0;f_x<3;++f_x) {
          int i_x = x+f_x-1;
          switch(border_type) {
            case conv_filter_border_type::BORDER_SKIP:
              // nothing to do
              break;
            case conv_filter_border_type::BORDER_ZERO:        // 000000|abcdefgh|0000000
              // nothing to do
              break;
            case conv_filter_border_type::BORDER_REPLICATE:   // aaaaaa|abcdefgh|hhhhhhh
              i_x = std::min(std::max(i_x,0),x_max);
              break;
            case conv_filter_border_type::BORDER_REFLECT:     // fedcba|abcdefgh|hgfedcb
              if(i_x < x_min) {
                i_x = -1*i_x - 1;
              } else if(i_x > x_max) {
                i_x = x_max - (i_x-width_px);
              }
              break;
            case conv_filter_border_type::BORDER_WRAP:        // cdefgh|abcdefgh|abcdefg
              if(i_x < 0) {
                i_x += width_px;
              } else if(i_x > x_max) {
                i_x -= width_px;
              }
              break;
            case conv_filter_border_type::BORDER_REFLECT_101: // gfedcb|abcdefgh|gfedcba
              if(i_x < 0) {
                i_x *= -1;
              } else if(i_x > x_max) {
                i_x = 2*x_max - i_x;
              }
              break;
          }
          // handling of BORDER_ZERO
          accu += ((i_y < 0 || i_y >= (height_px) || i_x < 0 || i_x >= (width_px)) ? 0x00 : h_image_in[i_y*input_pitch + i_x])*filter.d[f_y*3 + f_x];
        }
      }
      accu *= scale;
      h_image_out[y*output_pitch + x] = static_cast<unsigned char>(std::min(std::max(accu,0.0f),255.0f));
    }
  }
}

std::ostream & operator<<(std::ostream & os, const vilib::conv_filter_border_type_t & border_type) {
  switch(border_type) {
    case conv_filter_border_type::BORDER_SKIP:
      os << "BORDER_SKIP";
      break;
    case conv_filter_border_type::BORDER_ZERO:
      os << "BORDER_ZERO";
      break;
    case conv_filter_border_type::BORDER_REPLICATE:
      os << "BORDER_REPLICATE";
      break;
    case conv_filter_border_type::BORDER_REFLECT:
      os << "BORDER_REFLECT";
      break;
    case conv_filter_border_type::BORDER_WRAP:
      os << "BORDER_WRAP";
      break;
    case conv_filter_border_type::BORDER_REFLECT_101:
      os << "BORDER_REFLECT_101";
      break;
  }
  return os;
}

INSTANTIATE_2D_SEPARABLE(unsigned char, float, unsigned char);
INSTANTIATE_2D_SEPARABLE(unsigned char, float, float);
INSTANTIATE_2D_SEPARABLE(float, float, float);

} // namespace vilib