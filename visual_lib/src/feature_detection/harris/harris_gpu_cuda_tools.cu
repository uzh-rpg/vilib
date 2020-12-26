/*
 * Harris/Shi-Tomasi feature detector utilities in CUDA
 * harris_gpu_cuda_tools.cu
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

#include "vilib/cuda_common.h"
#include "vilib/feature_detection/harris/harris_gpu_cuda_tools.h"

namespace vilib {

static void harris_gpu_find_best_params(const int & cols,
                                        int & threadX,
                                        int & threadY) {
  int t_best = 128;
  for(int t=t_best,idle_threads_best = t_best;t>16;t>>=1) {
    int idle_threads_now = ((cols + t-1)/t)*t - cols;
    if(idle_threads_best > idle_threads_now) {
      t_best = t;
      idle_threads_best = idle_threads_now;
    }
  }
  threadX = t_best;
  threadY = 128/t_best;
}

static void harris_gpu_find_best_vector_params(const int & cols,
                                               const int & input_pitch,
                                               const int & output_pitch,
                                               int & v,
                                               int & cols_vectorized,
                                               int & input_pitch_vectorized,
                                               int & output_pitch_vectorized,
                                               int & threadX,
                                               int & threadY) {
  // Select the most efficient vectorized version
  v=4;
  for(;v>0;v>>=1) {
    int bitfield = v-1;
    if(((cols & bitfield) != 0) ||
       ((input_pitch & bitfield) != 0) ||
       ((output_pitch & bitfield) != 0)) {
         continue;
    }
    break;
  }
  cols_vectorized = cols/v;
  input_pitch_vectorized = input_pitch/v;
  output_pitch_vectorized = output_pitch/v;
  // Select a good thread block size that minimizes the number of idle threads
  int t_best = 128;
  for(int t=t_best,idle_threads_best = t_best;t>16;t>>=1) {
    int idle_threads_now = ((cols_vectorized + t-1)/t)*t - cols_vectorized;
    if(idle_threads_best > idle_threads_now) {
      t_best = t;
      idle_threads_best = idle_threads_now;
    }
  }
  threadX = t_best;
  threadY = 128/t_best;
}

template <int N>
__global__ void array_multiply_kernel(const float * __restrict__ a,
                                      const float * __restrict__ b,
                                      float * result,
                                      const int cols,
                                      const int rows,
                                      const int common_pitch) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if(ty < rows && tx < cols) {
    const int idx = ty * common_pitch + tx;
    if(N==1) {
      result[idx] = a[idx] * b[idx];
    } else if(N==2) {
      float2 * result_v = reinterpret_cast<float2*>(result);
      const float2 * a_v = reinterpret_cast<const float2*>(a);
      const float2 * b_v = reinterpret_cast<const float2*>(b);
      result_v[idx].x = a_v[idx].x * b_v[idx].x;
      result_v[idx].y = a_v[idx].y * b_v[idx].y;
    } else if(N==4) {
      float4 * result_v = reinterpret_cast<float4*>(result);
      const float4 * a_v = reinterpret_cast<const float4*>(a);
      const float4 * b_v = reinterpret_cast<const float4*>(b);
      result_v[idx].x = a_v[idx].x * b_v[idx].x;
      result_v[idx].y = a_v[idx].y * b_v[idx].y;
      result_v[idx].z = a_v[idx].z * b_v[idx].z;
      result_v[idx].w = a_v[idx].w * b_v[idx].w;
    }
  }
}

__host__ void harris_gpu_array_multiply(const float * d_input_a,
                                        const int input_a_pitch,
                                        const float * d_input_b,
                                        const int input_b_pitch,
                                        float * d_output,
                                        const int output_pitch,
                                        const int cols,
                                        const int rows,
                                        cudaStream_t stream) {
  if(input_a_pitch != input_b_pitch ||
     input_a_pitch != output_pitch) {
    throw std::runtime_error("Currently only common pitch is supported.");
  }
  int v, cols_vectorized, common_pitch_vectorized;
  int threadX, threadY;
  harris_gpu_find_best_vector_params(cols,
                                     input_a_pitch,
                                     input_a_pitch,
                                     v,
                                     cols_vectorized,
                                     common_pitch_vectorized,
                                     common_pitch_vectorized,
                                     threadX,
                                     threadY);
  kernel_params_t p = cuda_gen_kernel_params_2d(cols_vectorized,rows,threadX,threadY);
  decltype(&array_multiply_kernel<1>) kernel;
  switch(v) {
    case 4:
      kernel = array_multiply_kernel<4>;
      break;
    case 2:
      kernel = array_multiply_kernel<2>;
      break;
    case 1:
      kernel = array_multiply_kernel<1>;
      break;
    default:
      assert(0);
      kernel = array_multiply_kernel<1>;
      break;
  }
  kernel <<< p.blocks_per_grid, p.threads_per_block,0,stream>>>(
    d_input_a,
    d_input_b,
    d_output,
    cols_vectorized,
    rows,
    common_pitch_vectorized);
  CUDA_KERNEL_CHECK();
}

__inline__ __device__ void sum_neighbors(const float * __restrict__ d_dx2,
                                         const float * __restrict__ d_dy2,
                                         const float * __restrict__ d_dxdy,
                                         const int & pitch_elements,
                                         const int & col,
                                         const int & row,
                                         float & a,
                                         float & b,
                                         float & c) {
  a = 0.0f;
  b = 0.0f;
  c = 0.0f;
  const int start_offset = (row-1) * pitch_elements + (col-1);
  const int row_offset = (pitch_elements-3);
  const float * ptr_a = d_dx2 + start_offset;
  const float * ptr_b = d_dxdy + start_offset;
  const float * ptr_c = d_dy2 + start_offset;
  #pragma unroll 3
  for(int i=-1; i<2; ++i) {
    #pragma unroll 3
    for(int j=-1; j<2; ++j) {
      a += *ptr_a;
      b += *ptr_b;
      c += *ptr_c;
      ++ptr_a;
      ++ptr_b;
      ++ptr_c;
    }
    ptr_a += row_offset;
    ptr_b += row_offset;
    ptr_c += row_offset;
  }
  a *= 1.0f/9.0f;
  b *= 1.0f/9.0f;
  c *= 1.0f/9.0f;
}

template<bool use_harris>
__global__ void harris_gpu_calc_corner_response_kernel(
                                              const float * __restrict__ d_dx2,
                                              const float * __restrict__ d_dy2,
                                              const float * __restrict__ d_dxdy,
                                              const int input_pitch,
                                              float * __restrict__ d_response,
                                              const int output_pitch,
                                              const int minX,
                                              const int maxX,
                                              const int minY,
                                              const int maxY,
                                              const float k) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(y <= maxY && x <= maxX && y >= minY && x >= minX) {
    /*  
     * Notes to future self:
     * M = | dx2_int dxy_int | = | a b |
     *     | dxy_int dy2_int |   | b c |
     *
     * Run a 3x3 box filter
     */
    float a, b, c;
    sum_neighbors(d_dx2, d_dy2, d_dxdy, input_pitch, x, y, a, b, c);

    float r;
    if(use_harris) {
      // Harris score
      r = a*c -b*b - k*(a+c)*(a+c);
    } else {
      // Shi-Tomasi score: min(l1,l2)
      /*
       * l1 + l2 = a + c        -> l1 = a + c - l2
       * l1 * l2 = a * c - b^2  -> 0  = l2^2 - (a+c)l2 + (a*c-b^2)
       * l2 = (a+c +- sqrt((a+c)^2 - 4(a*c-b^2)) / 2
       * After simplification:
       * l2 = a/2 + c/2  +- sqrt((a/2 - c/2)^2 + b^2)
       * l1 = a/2 + c/2  -+ sqrt((a/2 - c/2)^2 + b^2)
       *
       * Remark:
       * We multiplied the response by 2 for less arithmetics:
       * This will introduce a 2x scale difference between a "true"
       * implementation, and this.
       */
      r = (a+c) - sqrtf((a-c)*(a-c) + 4*b*b);
    }
    d_response[y * output_pitch + x] = r;
  }
}

__host__ void harris_gpu_calc_corner_response(const float * d_dx2,
                                              const int dx2_pitch,
                                              const float * d_dy2,
                                              const int dy2_pitch,
                                              const float * d_dxdy,
                                              const int dxdy_pitch,
                                              float * d_response,
                                              const int response_pitch,
                                              const int cols,
                                              const int rows,
                                              const conv_filter_border_type_t border_type,
                                              const bool use_harris,
                                              const float k,
                                              cudaStream_t stream) {
  if(dx2_pitch != dy2_pitch ||
     dx2_pitch != dxdy_pitch) {
    throw std::runtime_error("Currently only common pitch is supported.");
  }
  int threadX,threadY;
  harris_gpu_find_best_params(cols,threadX,threadY);
  kernel_params_t p = cuda_gen_kernel_params_2d(cols,rows,threadX,threadY);
  const int minX = 1 + (border_type == conv_filter_border_type::BORDER_SKIP?1:0);
  const int maxX = (cols-2) - (border_type == conv_filter_border_type::BORDER_SKIP?1:0);
  const int minY = 1 + (border_type == conv_filter_border_type::BORDER_SKIP?1:0);
  const int maxY = (rows-2) - (border_type == conv_filter_border_type::BORDER_SKIP?1:0);
  decltype(&harris_gpu_calc_corner_response_kernel<false>) kernel;
  if(use_harris) {
    kernel = harris_gpu_calc_corner_response_kernel<true>;
  } else {
    kernel = harris_gpu_calc_corner_response_kernel<false>;
  }
  kernel <<< p.blocks_per_grid, p.threads_per_block,0,stream>>> (
    d_dx2,
    d_dy2,
    d_dxdy,
    dx2_pitch,
    d_response,
    response_pitch,
    minX,
    maxX,
    minY,
    maxY,
    k);
  CUDA_KERNEL_CHECK();
}

} // namespace vilib