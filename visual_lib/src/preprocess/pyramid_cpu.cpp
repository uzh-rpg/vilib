/*
 * Functions for creating image pyramids on the CPU
 * pyramid_cpu.cpp
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

#include <assert.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vilib/preprocess/pyramid.h"
//CPU Architecture specific includes
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define PYRAMID_SIMD_ENABLE         1
/*
 * Note to future self:
 * the SSE avergage instructions introduce a +1 in the numerator,
 * making the vectorized version's output different from non-vectorized
 * implementation. When using strict conformance, we do averaging using
 * a simple shift operation instead of the avg instruction.
 */
#define STRICT_CONFORMANCE          1

namespace vilib {

static void image_halfsample_cpu(const cv::Mat & in, cv::Mat & out);
#if PYRAMID_SIMD_ENABLE
#if defined(__SSE2__)
static void image_halfsample16_cpu_sse2(const cv::Mat & in, cv::Mat & out);
static inline bool is_address_aligned16(const void * ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0F) == 0x00;
}
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
static void image_halfsample_cpu_neon(const cv::Mat & in, cv::Mat & out);
#endif /* defined(__SSE2__) || defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

void pyramid_create_cpu(const cv::Mat & h_image_in,
                        std::vector<cv::Mat> & h_image_pyramid,
                        unsigned int levels,
                        bool deep_copy) {
  assert(h_image_in.type() == CV_8U);
  assert(h_image_in.rows > 0);
  assert(h_image_in.cols > 0);
  assert(levels > 0);

  h_image_pyramid.resize(levels);
  if(deep_copy) {
    h_image_pyramid[0] = h_image_in.clone();
  } else {
    h_image_pyramid[0] = h_image_in;
  }
  for(unsigned int i=1;i<levels;++i) {
    h_image_pyramid[i] = cv::Mat(h_image_pyramid[i-1].rows/2,
                                 h_image_pyramid[i-1].cols/2,
                                 CV_8U);
    image_halfsample_cpu(h_image_pyramid[i-1],h_image_pyramid[i]);
  }
}

void pyramid_create_cpu(std::vector<cv::Mat> & h_image_pyramid) {
  assert(h_image_pyramid[0].type() == CV_8U);
  assert(h_image_pyramid[0].rows > 0);
  assert(h_image_pyramid[0].cols > 0);
  assert(h_image_pyramid.size() > 1);

  for(unsigned int i=1;i<h_image_pyramid.size();++i) {
    h_image_pyramid[i] = cv::Mat(h_image_pyramid[i-1].rows/2,
                                 h_image_pyramid[i-1].cols/2,
                                 CV_8U);
    image_halfsample_cpu(h_image_pyramid[i-1],h_image_pyramid[i]);
  }
}

static void image_halfsample_cpu(const cv::Mat & in, cv::Mat & out) {
  assert(in.rows/2 == out.rows);
  assert(in.cols/2 == out.cols);
#if PYRAMID_SIMD_ENABLE
#if defined(__SSE2__)
  if(is_address_aligned16(in.data) &&
     is_address_aligned16(out.data) &&
     ((in.cols % 16) == 0) &&
     static_cast<std::size_t>(in.cols) == in.step) {
    image_halfsample16_cpu_sse2(in,out);
    return;
  }
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
  if((in.cols % 16) == 0) {
    image_halfsample_cpu_neon(in,out);
    return;
  }
#endif /* defined(__SSE2__) || defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

  const int in_stride = in.step.p[0];
  const int out_stride = out.step.p[0];
  uint8_t* top = reinterpret_cast<uint8_t*>(in.data);
  uint8_t* bottom = top + in_stride;
  uint8_t* end = top + in_stride*in.rows;
  uint8_t* p = reinterpret_cast<uint8_t*>(out.data);
  for (int y=0; y < out.rows && bottom < end; y++, top += in_stride*2, bottom += in_stride*2, p += out_stride)
  {
    for (int x=0; x < out.cols; x++)
    {
       p[x] = static_cast<uint8_t>((static_cast<uint16_t>(top[x*2]) + top[x*2+1] + bottom[x*2] + bottom[x*2+1])/4 );
    }
  }
}

#if PYRAMID_SIMD_ENABLE
#ifdef __SSE2__
static void image_halfsample16_cpu_sse2(const cv::Mat & in, cv::Mat & out) {
  const unsigned char * in_data = in.data;
  unsigned char * out_data = out.data;
  const unsigned char* nextRow = in_data + in.cols;
  const int w = in.cols;
  const int h = in.rows;
  int sw = w >> 4;
  int sh = h >> 1;
  __m128i m = _mm_set1_epi16(0x00FF);
  for (int i=0; i<sh; i++) {
    for (int j=0; j<sw; j++) {
      __m128i here = _mm_load_si128(reinterpret_cast<const __m128i*>(in_data));
      __m128i next = _mm_load_si128(reinterpret_cast<const __m128i*>(nextRow));
#if STRICT_CONFORMANCE
      here = _mm_add_epi16(_mm_and_si128(here,m),_mm_and_si128(_mm_srli_si128(here,1), m));
      next = _mm_add_epi16(_mm_and_si128(next,m),_mm_and_si128(_mm_srli_si128(next,1), m));
      here = _mm_srli_epi16(_mm_add_epi16(here,next),2);
#else
      here = _mm_avg_epu8(here,next);
      next = _mm_and_si128(_mm_srli_si128(here,1), m);
      here = _mm_and_si128(here,m); 
      here = _mm_avg_epu16(here, next);
#endif /* STRICT_CONFORMANCE */
      /*
       * Note to future self:
       * with GCC-7, the proper unaligned store is not available.
       */
      _mm_storel_epi64((__m128i*)out_data, _mm_packus_epi16(here,here));
      in_data += 16;
      nextRow += 16;
      out_data += 8;
    }
    in_data += w;
    nextRow += w;
  }
}
#endif /* __SSE2__ */

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
void image_halfsample_cpu_neon(const cv::Mat & in, cv::Mat & out) {
  const int in_stride = in.step.p[0];
  const int out_stride = out.step.p[0];
  for (int y = 0; y < in.rows; y += 2) {
    const uint8_t * in_top = in.data + y*in_stride;
    const uint8_t * in_bottom = in.data + (y+1)*in_stride;
    uint8_t * out_data = out.data + (y >> 1)*out_stride;
    for (int x = in.cols; x > 0 ; x-=16, in_top += 16, in_bottom += 16, out_data += 8) {
      uint8x8x2_t top  = vld2_u8( (const uint8_t *)in_top );
      uint8x8x2_t bottom = vld2_u8( (const uint8_t *)in_bottom );
      uint16x8_t sum = vaddl_u8( top.val[0], top.val[1] );
      sum = vaddw_u8( sum, bottom.val[0] );
      sum = vaddw_u8( sum, bottom.val[1] );
      uint8x8_t final_sum = vshrn_n_u16(sum, 2);
      vst1_u8(out_data, final_sum);
    }
  }
}
#endif /* defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

void pyramid_display(const std::vector<cv::Mat> & pyramid) {
  for(std::size_t l=0;l<pyramid.size();++l) {
    std::string subframe_title("Image pyramid (");
    subframe_title += std::to_string(pyramid[l].cols);
    subframe_title += "x";
    subframe_title += std::to_string(pyramid[l].rows);
    subframe_title += ")";
    cv::imshow(subframe_title.c_str(), pyramid[l]);
    cv::waitKey();
  }
}

} // namespace vilib