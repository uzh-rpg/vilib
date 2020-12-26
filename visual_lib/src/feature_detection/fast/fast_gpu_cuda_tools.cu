/*
 * FAST feature detector utilities in CUDA
 * fast_gpu_cuda_tools.cu
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

#include "vilib/feature_detection/config.h"
#include "vilib/feature_detection/fast/fast_gpu_cuda_tools.h"
#include "vilib/cuda_common.h"

namespace vilib {

__inline__ __device__ int bresenham_circle_offset_pitch(const int & i,
                                                        const int & pitch,
                                                        const int & pitch2,
                                                        const int & pitch3) {
  /*
   * Note to future self and others:
   * this function is only should be called in for loops that are unrolled.
   * Due to unrollment, the if else structure disappears, and the offsets get
   * substituted.
   *
   * Order within the circle:
   *
   *      7 8  9
   *    6       10
   *  5           11
   *  4     x     12
   *  3           13
   *   2        14
   *     1 0 15
   */
  int offs=0;
  if(i==0)
    offs=pitch3;
  else if(i==1)
    offs=pitch3-1;
  else if(i==2)
    offs=pitch2-2;
  else if(i==3)
    offs=pitch-3;
  else if(i==4)
    offs=-3;
  else if(i==5)
    offs=-pitch-3;
  else if(i==6)
    offs=-pitch2-2;
  else if(i==7)
    offs=-pitch3-1;
  else if(i==8)
    offs=-pitch3;
  else if(i==9)
    offs=-pitch3+1;
  else if(i==10)
    offs=-pitch2+2;
  else if(i==11)
    offs=-pitch+3;
  else if(i==12)
    offs=3;
  else if(i==13)
    offs=pitch+3;
  else if(i==14)
    offs=pitch2+2;
  else if(i==15)
    offs=pitch3+1;
  return offs;
}

__inline__ __device__ unsigned char fast_gpu_is_corner(const unsigned int & address,
                                                       const int & min_arc_length) {
  int ones = __popc(address);
  if(ones < min_arc_length) { // if we dont have enough 1-s in the address, dont even try
    return 0;
  }
  unsigned int address_dup = address|(address<<16); //duplicate the low 16-bits at the high 16-bits
  while(ones > 0) {
    address_dup <<= __clz(address_dup); // shift out the high order zeros
    int lones = __clz(~address_dup); // count the leading ones
    if(lones >= min_arc_length) {
      return 1;
    }
    address_dup <<= lones; // shift out the high order ones
    ones -= lones;
  }
  return 0;
}

__inline __device__ unsigned int fast_gpu_prechecks(const float & c_t,
                                                    const float & ct,
                                                    const unsigned char * image_ptr,
                                                    const int & image_pitch,
                                                    const int & image_pitch2,
                                                    const int & image_pitch3) {
  /*
   * Note to future self:
   * using too many prechecks of course doesnt help
   */
  // (-3,0) (3,0) -> 4,12
  float px0 = (float)image_ptr[bresenham_circle_offset_pitch(4,image_pitch,image_pitch2,image_pitch3)];
  float px1 = (float)image_ptr[bresenham_circle_offset_pitch(12,image_pitch,image_pitch2,image_pitch3)];
  if((signbit(px0-c_t)|signbit(px1-c_t)|signbit(ct-px0)|signbit(ct-px1))==0) {
    return 1;
  }
  // (0,3), (0,-3) -> 0, 8
  px0 = (float)image_ptr[bresenham_circle_offset_pitch(0,image_pitch,image_pitch2,image_pitch3)];
  px1 = (float)image_ptr[bresenham_circle_offset_pitch(8,image_pitch,image_pitch2,image_pitch3)];
  if((signbit(px0-c_t)|signbit(px1-c_t)|signbit(ct-px0)|signbit(ct-px1))==0) {
    return 1;
  }
  return 0;
}

#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
__global__ void fast_gpu_calculate_lut_kernel(unsigned int * __restrict__ d_corner_lut,
                                              const int min_arc_length) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // 11 bits come from here
  unsigned int output = 0;

  #pragma unroll
  for (int i=0;i<32;++i) {
    // the top 5 bits come from here
    // in total, 2^16 combinations
    const int unified_address = (i<<11) | x;
    output |= fast_gpu_is_corner(unified_address,min_arc_length)<<i;
  }
  d_corner_lut[x] = output;
}

__host__ void fast_gpu_calculate_lut(unsigned int * d_corner_lut,
                                     const int & min_arc_length,
                                     cudaStream_t stream) {
  // every thread writes 4 bytes: in total 64kbits get written
  kernel_params_t p = cuda_gen_kernel_params_1d(2048,256);
  fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid,p.threads_per_block,0,stream>>>
                                                   ((unsigned int*)d_corner_lut,
                                                    min_arc_length);
  CUDA_KERNEL_CHECK();
}
#else
__global__ void fast_gpu_calculate_lut_kernel(unsigned char * __restrict__ d_corner_lut,
                                              const int min_arc_length) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // all 16 bits come from here
  d_corner_lut[x] = fast_gpu_is_corner(x,min_arc_length);
}

__host__ void fast_gpu_calculate_lut(unsigned char * d_corner_lut,
                                     const int & min_arc_length,
                                     cudaStream_t stream) {
  // every thread writes a byte: in total 64kB gets written
  kernel_params_t p = cuda_gen_kernel_params_1d(64*1024,256);
  fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid,p.threads_per_block,0,stream>>>
                                                   (d_corner_lut,
                                                    min_arc_length);
  CUDA_KERNEL_CHECK();
}
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

__inline__ __device__ int fast_gpu_is_corner_quick(
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                  const unsigned int * __restrict__ d_corner_lut,
#else
                                  const unsigned char * __restrict__ d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                  const float * __restrict__ px,
                                  const float & center_value,
                                  const float & threshold,
                                  unsigned int & dark_diff_address,
                                  unsigned int & bright_diff_address
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                  ,
                                  unsigned int & dark_diff_bit,
                                  unsigned int & dark_diff_data,
                                  unsigned int & bright_diff_bit,
                                  unsigned int & bright_diff_data
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                ) {
  const float ct = center_value + threshold;
  const float c_t = center_value - threshold;
  dark_diff_address = 0;
  bright_diff_address = 0;

#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  dark_diff_bit = 0;
  bright_diff_bit = 0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

  #pragma unroll 16
  for(int i=0;i<16;++i) {
    int darker = signbit(px[i]-c_t);
    int brighter = signbit(ct-px[i]);
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
    if(i<11) {
      dark_diff_address   += darker?(1<<i):0;
      bright_diff_address += brighter?(1<<i):0;
    } else {
      if(i==11) {
        // initiate a readout
        dark_diff_data = d_corner_lut[dark_diff_address];
        bright_diff_data = d_corner_lut[bright_diff_address];
      }
      dark_diff_bit   += darker?(1<<(i-11)):0;
      bright_diff_bit += brighter?(1<<(i-11)):0;
    }
#else
    dark_diff_address   += signbit(px[i]-c_t)?(1<<i):0;
    bright_diff_address += signbit(ct-px[i])?(1<<i):0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
  }
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  return (dark_diff_data&(1<<dark_diff_bit)) | (bright_diff_data&(1<<bright_diff_bit));
#else
  return (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]);
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
}

template <fast_score SCORE>
__global__ void fast_gpu_calc_corner_response_kernel(
                                       const int image_width,
                                       const int image_height,
                                       const int image_pitch,
                                       const unsigned char * __restrict__ d_image,
                                       const int horizontal_border,
                                       const int vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                       const unsigned int * __restrict__ d_corner_lut,
#else
                                       const unsigned char * __restrict__ d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                       const float threshold,
                                       const int min_arc_length,
                                       const int response_pitch_elements,
                                       float * __restrict__ d_response) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // thread id X
  const int y = blockDim.y*blockIdx.y + threadIdx.y; // thread id Y
  if(x < image_width && y < image_height) {
    const int resp_offset = y*response_pitch_elements + x;
    d_response[resp_offset] = 0.0f;
    if((x >= horizontal_border) &&
       (y >= vertical_border) &&
       (x < (image_width - horizontal_border)) &&
       (y < (image_height - vertical_border))) {
      const unsigned char * d_image_ptr = d_image + y*image_pitch + x;
      const float c = (float)(*d_image_ptr);
      const float ct = c + threshold;
      const float c_t = c - threshold;
      /*
       * Note to future self:
       * we need to create 2 differences for each of the 16 pixels
       * have 1 lookup table, and look-up both values
       *
       * c_t stands for: c - threshold (epsilon)
       * ct stands for : c + threshold (epsilon)
       *
       * Label of px:
       * - darker  if   px < c_t              (1)
       * - similar if   c_t <= px <= ct      (2)
       * - brighter if  ct < px             (3)
       *
       * Darker diff: px - c_t
       * sign will only give 1 in case of (1), and 0 in case of (2),(3)
       *
       * Similarly, brighter diff: ct - px
       * sign will only give 1 in case of (3), and 0 in case of (2),(3)
       */
      unsigned int dark_diff_address = 0;
      unsigned int bright_diff_address = 0;
#if FAST_GPU_USE_LOOKUP_TABLE && FAST_GPU_USE_LOOKUP_TABLE_BITBASED
      unsigned int dark_diff_bit = 0;
      unsigned int bright_diff_bit = 0;
      unsigned int dark_diff_data = 0;
      unsigned int bright_diff_data = 0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

      // Precalculate pitches
      const int image_pitch2 = image_pitch << 1;
      const int image_pitch3 = image_pitch + (image_pitch << 1);

      // Do a coarse corner check
      // TODO: I could use the results of the prechecks afterwards
      if(fast_gpu_prechecks(c_t,ct,d_image_ptr,image_pitch,image_pitch2,image_pitch3)) {
        return;
      }

      float px[16];
      #pragma unroll 16
      for(int i=0;i<16;++i) {
        int image_ptr_offset = bresenham_circle_offset_pitch(i,image_pitch,image_pitch2,image_pitch3);
        px[i] = (float)d_image_ptr[image_ptr_offset];
        int darker = signbit(px[i]-c_t);
        int brighter = signbit(ct-px[i]);
#if FAST_GPU_USE_LOOKUP_TABLE && FAST_GPU_USE_LOOKUP_TABLE_BITBASED
        if(i<11) {
          dark_diff_address   += darker?(1<<i):0;
          bright_diff_address += brighter?(1<<i):0;
        } else {
          if(i==11) {
            // initiate a readout
            dark_diff_data = d_corner_lut[dark_diff_address];
            bright_diff_data = d_corner_lut[bright_diff_address];
          }
          dark_diff_bit   += darker?(1<<(i-11)):0;
          bright_diff_bit += brighter?(1<<(i-11)):0;
        }
#else
        dark_diff_address   += signbit(px[i]-c_t)?(1<<i):0;
        bright_diff_address += signbit(ct-px[i])?(1<<i):0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
      }
      // Look up these addresses, whether they qualify for a corner
      // If any of these qualify for a corner, it is a corner candidate, yaay
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
      if((dark_diff_data&(1<<dark_diff_bit)) || (bright_diff_data&(1<<bright_diff_bit))) {
#else
      if(d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]) {
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#else
      if(fast_gpu_is_corner(dark_diff_address,min_arc_length) ||
         fast_gpu_is_corner(bright_diff_address,min_arc_length)) {
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        /*
         * Note to future self:
         * Only calculate the score once we determined that the pixel is considered
         * a corner. This policy gave better results than computing the score
         * for every pixel
         */
        if(SCORE == SUM_OF_ABS_DIFF_ALL) {
          float response = 0.0f;
          #pragma unroll 16
          for(int i=0;i<16;++i) {
            response += fabsf(px[i]-c);
          }
          d_response[resp_offset] = response;
        } else if(SCORE == SUM_OF_ABS_DIFF_ON_ARC) {
          float response_bright = 0.0f;
          float response_dark   = 0.0f;
          #pragma unroll 16
          for(int i=0;i<16;++i) {
            float absdiff = fabsf(px[i]-c)-threshold;
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
            if(i<11) {
              // address
              response_dark += (dark_diff_address&(1<<i))?absdiff:0.0f;
              response_bright += (bright_diff_address&(1<<i))?absdiff:0.0f;
            } else {
              // bits
              response_dark += (dark_diff_bit&(1<<(i-11)))?absdiff:0.0f;
              response_bright += (bright_diff_bit&(1<<(i-11)))?absdiff:0.0f;
            }
#else
            response_dark   += (dark_diff_address&(1<<i))?absdiff:0.0f;
            response_bright += (bright_diff_address&(1<<i))?absdiff:0.0f;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
          }
          d_response[resp_offset] = fmaxf(response_bright,response_dark);
        } else if(SCORE == MAX_THRESHOLD) {
          // Binary search for the maximum threshold value with which the given
          // point is still a corner
          float min_thr = threshold + 1;
          float max_thr = 255.0f;
          while(min_thr <= max_thr) {
            float med_thr = floorf((min_thr + max_thr)*0.5f);
            // try out med_thr as a new threshold
            if(fast_gpu_is_corner_quick(d_corner_lut,
                                        px,
                                        c,
                                        med_thr,
                                        dark_diff_address,
                                        bright_diff_address
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                        ,
                                        dark_diff_bit,
                                        dark_diff_data,
                                        bright_diff_bit,
                                        bright_diff_data
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                      )) {
              // still a corner
              min_thr = med_thr + 1.0f;
            } else {
              // not a corner anymore
              max_thr = med_thr - 1.0f;
            }
          }
          d_response[resp_offset] = max_thr;
        }
      }
    }
  }
}

__host__ void fast_gpu_calc_corner_response(const int image_width,
                                            const int image_height,
                                            const int image_pitch,
                                            const unsigned char * d_image,
                                            const int horizontal_border,
                                            const int vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                            const unsigned int * d_corner_lut,
#else
                                            const unsigned char * d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                            const float threshold,
                                            const int min_arc_length,
                                            const fast_score score,
                                            const int response_pitch_elements,
                                            float * d_response,
                                            cudaStream_t stream) {
  // Note: I'd like to launch 128 threads / thread block
  std::size_t threads_per_x = (image_width%32 == 0)?32:16;
  std::size_t threads_per_y = 128/threads_per_x;
  kernel_params_t p = cuda_gen_kernel_params_2d(image_width,
                                                image_height,
                                                threads_per_x,
                                                threads_per_y);
  switch(score) {
    case SUM_OF_ABS_DIFF_ALL:
      fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ALL> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
          image_width,
          image_height,
          image_pitch,
          d_image,
          horizontal_border,
          vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
          d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
          threshold,
          min_arc_length,
          response_pitch_elements,
          d_response
        );
      break;
    case SUM_OF_ABS_DIFF_ON_ARC:
      fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ON_ARC> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
        image_width,
        image_height,
        image_pitch,
        d_image,
        horizontal_border,
        vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
        d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        threshold,
        min_arc_length,
        response_pitch_elements,
        d_response
      );
      break;
    case MAX_THRESHOLD:
      fast_gpu_calc_corner_response_kernel<MAX_THRESHOLD> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
        image_width,
        image_height,
        image_pitch,
        d_image,
        horizontal_border,
        vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
        d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        threshold,
        min_arc_length,
        response_pitch_elements,
        d_response
      );
      break;
  }
  CUDA_KERNEL_CHECK();
}

} // namespace vilib
