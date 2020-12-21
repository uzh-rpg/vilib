/*
 * Detector Base GPU CUDA tools
 * detector_base_gpu_cuda_tools.cu
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

#include "vilib/feature_detection/detector_base_gpu_cuda_tools.h"
#include "vilib/feature_detection/config.h"
#include "vilib/cuda_common.h"

namespace vilib {

  // Warp preliminaries
#define WARP_MASK                            0xFFFFFFFF
// Use precalculated indices
/*
 * Note to future self: this is just a coding convenience,
 * because the loop is unrolled, there's no speed impact
 */
#define USE_PRECALCULATED_INDICES            1

#if USE_PRECALCULATED_INDICES
__inline__ __device__ void nms_offset_pos(const int & i, float & x_offset, float & y_offset) {
  /*
   * 3x3 (n=1):
   * 7 0 1
   * 6 x 2
   * 5 4 3
   */
   if(i==0) {
     x_offset = 0.0f;
     y_offset = -1.0f;
   }
   if(i==1) {
     x_offset = 1.0f;
     y_offset = -1.0f;
   }
   if(i==2) {
     x_offset = 1.0f;
     y_offset = 0.0f;
   }
   if(i==3) {
     x_offset = 1.0f;
     y_offset = 1.0f;
   }
   if(i==4) {
     x_offset = 0.0f;
     y_offset = 1.0f;
   }
   if(i==5) {
     x_offset = -1.0f;
     y_offset = 1.0f;
   }
   if(i==6) {
     x_offset = -1.0f;
     y_offset = 0.0f;
   }
   if(i==7) {
     x_offset = -1.0f;
     y_offset = -1.0f;
   }
 #if (DETECTOR_BASE_NMS_SIZE >= 5)
  /*
   * 5x5 (n=2):
   * 23 8  9  10 11
   * 22 7  0  1  12
   * 21 6  x  2  13
   * 20 5  4  3  14
   * 19 18 17 16 15
   */
   if(i==8) {
     x_offset = -1.0f;
     y_offset = -2.0f;
   }
   if(i==9) {
     x_offset = 0.0f;
     y_offset = -2.0f;
   }
   if(i==10) {
     x_offset = 1.0f;
     y_offset = -2.0f;
   }
   if(i==11) {
     x_offset = 2.0f;
     y_offset = -2.0f;
   }
   if(i==12) {
     x_offset = 2.0f;
     y_offset = -1.0f;
   }
   if(i==13) {
     x_offset = 2.0f;
     y_offset = 0.0f;
   }
   if(i==14) {
     x_offset = 2.0f;
     y_offset = 1.0f;
   }
   if(i==15) {
     x_offset = 2.0f;
     y_offset = 2.0f;
   }
   if(i==16) {
     x_offset = 1.0f;
     y_offset = 2.0f;
   }
   if(i==17) {
     x_offset = 0.0f;
     y_offset = 2.0f;
   }
   if(i==18) {
     x_offset = -1.0f;
     y_offset = 2.0f;
   }
   if(i==19) {
     x_offset = -2.0f;
     y_offset = 2.0f;
   }
   if(i==20) {
     x_offset = -2.0f;
     y_offset = 1.0f;
   }
   if(i==21) {
     x_offset = -2.0f;
     y_offset = 0.0f;
   }
   if(i==22) {
     x_offset = -2.0f;
     y_offset = -1.0f;
   }
   if(i==23) {
     x_offset = -2.0f;
     y_offset = -2.0f;
   }
#if (DETECTOR_BASE_NMS_SIZE >= 7)
  /*
   * 7x7 (n=3)
   * 47 24 25 26 27 28 29
   * 46 23 8  9  10 11 30
   * 45 22 7  0  1  12 31
   * 44 21 6  x  2  13 32
   * 43 20 5  4  3  14 33
   * 42 19 18 17 16 15 34
   * 41 40 39 38 37 36 35
   */
   if(i==24) {
     x_offset = -2.0f;
     y_offset = -3.0f;
   }
   if(i==25) {
     x_offset = -1.0f;
     y_offset = -3.0f;
   }
   if(i==26) {
    x_offset = 0.0f;
    y_offset = -3.0f;
  }
  if(i==27) {
    x_offset = 1.0f;
    y_offset = -3.0f;
  }
  if(i==28) {
    x_offset = 2.0f;
    y_offset = -3.0f;
  }
  if(i==29) {
    x_offset = 3.0f;
    y_offset = -3.0f;
  }
  if(i==30) {
    x_offset = 3.0f;
    y_offset = -2.0f;
  }
  if(i==31) {
    x_offset = 3.0f;
    y_offset = -1.0f;
  }
  if(i==32) {
    x_offset = 3.0f;
    y_offset = 0.0f;
  }
  if(i==33) {
    x_offset = 3.0f;
    y_offset = 1.0f;
  }
  if(i==34) {
    x_offset = 3.0f;
    y_offset = 2.0f;
  }
  if(i==35) {
    x_offset = 3.0f;
    y_offset = 3.0f;
  }
  if(i==36) {
    x_offset = 2.0f;
    y_offset = 3.0f;
  }
  if(i==37) {
    x_offset = 1.0f;
    y_offset = 3.0f;
  }
  if(i==38) {
    x_offset = 0.0f;
    y_offset = 3.0f;
  }
  if(i==39) {
    x_offset = -1.0f;
    y_offset = 3.0f;
  }
  if(i==40) {
    x_offset = -2.0f;
    y_offset = 3.0f;
  }
  if(i==41) {
    x_offset = -3.0f;
    y_offset = 3.0f;
  }
  if(i==42) {
    x_offset = -3.0f;
    y_offset = 2.0f;
  }
  if(i==43) {
    x_offset = -3.0f;
    y_offset = 1.0f;
  }
  if(i==44) {
    x_offset = -3.0f;
    y_offset = 0.0f;
  }
  if(i==45) {
    x_offset = -3.0f;
    y_offset = -1.0f;
  }
  if(i==46) {
    x_offset = -3.0f;
    y_offset = -2.0f;
  }
  if(i==47) {
    x_offset = -3.0f;
    y_offset = -3.0f;
  }
#if (DETECTOR_BASE_NMS_SIZE >= 9)
  /* 
   * 9x9 (n=4)
   * 79 48 49 50 51 52 53 54 55
   * 78 47 24 25 26 27 28 29 56 
   * 77 46 23 8  9  10 11 30 57
   * 76 45 22 7  0  1  12 31 58
   * 75 44 21 6  x  2  13 32 59
   * 74 43 20 5  4  3  14 33 60
   * 73 42 19 18 17 16 15 34 61
   * 72 41 40 39 38 37 36 35 62
   * 71 70 69 68 67 66 65 64 63
   */
  if(i==48) {
    x_offset = -3.0f;
    y_offset = -4.0f;
  }
  if(i==49) {
    x_offset = -2.0f;
    y_offset = -4.0f;
  }
  if(i==50) {
    x_offset = -1.0f;
    y_offset = -4.0f;
  }
  if(i==51) {
    x_offset = 0.0f;
    y_offset = -4.0f;
  }
  if(i==52) {
    x_offset = 1.0f;
    y_offset = -4.0f;
  }
  if(i==53) {
    x_offset = 2.0f;
    y_offset = -4.0f;
  }
  if(i==54) {
    x_offset = 3.0f;
    y_offset = -4.0f;
  }
  if(i==55) {
    x_offset = 4.0f;
    y_offset = -4.0f;
  }
  if(i==56) {
    x_offset = 4.0f;
    y_offset = -3.0f;
  }
  if(i==57) {
    x_offset = 4.0f;
    y_offset = -2.0f;
  }
  if(i==58) {
    x_offset = 4.0f;
    y_offset = -1.0f;
  }
  if(i==59) {
    x_offset = 4.0f;
    y_offset = 0.0f;
  }
  if(i==60) {
    x_offset = 4.0f;
    y_offset = 1.0f;
  }
  if(i==61) {
    x_offset = 4.0f;
    y_offset = 2.0f;
  }
  if(i==62) {
    x_offset = 4.0f;
    y_offset = 3.0f;
  }
  if(i==63) {
    x_offset = 4.0f;
    y_offset = 4.0f;
  }
  if(i==64) {
    x_offset = 3.0f;
    y_offset = 4.0f;
  }
  if(i==65) {
    x_offset = 2.0f;
    y_offset = 4.0f;
  }
  if(i==66) {
    x_offset = 1.0f;
    y_offset = 4.0f;
  }
  if(i==67) {
    x_offset = 0.0f;
    y_offset = 4.0f;
  }
  if(i==68) {
    x_offset = -1.0f;
    y_offset = 4.0f;
  }
  if(i==69) {
    x_offset = -2.0f;
    y_offset = 4.0f;
  }
  if(i==70) {
    x_offset = -3.0f;
    y_offset = 4.0f;
  }
  if(i==71) {
    x_offset = -4.0f;
    y_offset = 4.0f;
  }
  if(i==72) {
    x_offset = -4.0f;
    y_offset = 3.0f;
  }
  if(i==73) {
    x_offset = -4.0f;
    y_offset = 2.0f;
  }
  if(i==74) {
    x_offset = -4.0f;
    y_offset = 1.0f;
  }
  if(i==75) {
    x_offset = -4.0f;
    y_offset = 0.0f;
  }
  if(i==76) {
    x_offset = -4.0f;
    y_offset = -1.0f;
  }
  if(i==77) {
    x_offset = -4.0f;
    y_offset = -2.0f;
  }
  if(i==78) {
    x_offset = -4.0f;
    y_offset = -3.0f;
  }
  if(i==79) {
    x_offset = -4.0f;
    y_offset = -4.0f;
  }
#endif /* (DETECTOR_BASE_NMS_SIZE >= 9) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 7) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 5) */
}

__inline__ __device__ int nms_offset(const int & i, const int & pitch) {
  int offs=0;
  /*
   * 3x3 (n=1):
   * 7 0 1
   * 6 x 2
   * 5 4 3
   */
  if(i==0)
    offs=-pitch;
  if(i==1)
    offs=-pitch+1;
  if(i==2)
    offs=+1;
  if(i==3)
    offs=pitch+1;
  if(i==4)
    offs=pitch;
  if(i==5)
    offs=pitch-1;
  if(i==6)
    offs= -1;
  if(i==7)
    offs=-pitch-1;
#if (DETECTOR_BASE_NMS_SIZE >= 5)
  /*
   * 5x5 (n=2):
   * 23 8  9  10 11
   * 22 7  0  1  12
   * 21 6  x  2  13
   * 20 5  4  3  14
   * 19 18 17 16 15
   */
  if(i==8)
    offs=-(pitch<<1)-1;
  if(i==9)
    offs=-(pitch<<1);
  if(i==10)
    offs=-(pitch<<1)+1;
  if(i==11)
    offs=-(pitch<<1)+2;
  if(i==12)
    offs=-pitch+2;
  if(i==13)
    offs=+2;
  if(i==14)
    offs=pitch+2;
  if(i==15)
    offs=(pitch<<1)+2;
  if(i==16)
    offs=(pitch<<1)+1;
  if(i==17)
    offs=(pitch<<1);
  if(i==18)
    offs=(pitch<<1)-1;
  if(i==19)
    offs=(pitch<<1)-2;
  if(i==20)
    offs=pitch-2;
  if(i==21)
    offs=-2;
  if(i==22)
    offs=-pitch-2;
  if(i==23)
    offs=-(pitch<<1)-2;
#if (DETECTOR_BASE_NMS_SIZE >= 7)
  /*
   * 7x7 (n=3)
   * 47 24 25 26 27 28 29
   * 46 23 8  9  10 11 30
   * 45 22 7  0  1  12 31
   * 44 21 6  x  2  13 32
   * 43 20 5  4  3  14 33
   * 42 19 18 17 16 15 34
   * 41 40 39 38 37 36 35
   */
  if(i==24)
    offs=-((pitch<<1)+pitch)-2;
  if(i==25)
    offs=-((pitch<<1)+pitch)-1;
  if(i==26)
    offs=-((pitch<<1)+pitch);
  if(i==27)
    offs=-((pitch<<1)+pitch)+1;
  if(i==28)
    offs=-((pitch<<1)+pitch)+2;
  if(i==29)
    offs=-((pitch<<1)+pitch)+3;
  if(i==30)
    offs=-(pitch<<1)+3;
  if(i==31)
    offs=-pitch+3;
  if(i==32)
    offs=3;
  if(i==33)
    offs=pitch+3;
  if(i==34)
    offs=(pitch<<1)+3;
  if(i==35)
    offs=((pitch<<1)+pitch)+3;
  if(i==36)
    offs=((pitch<<1)+pitch)+2;
  if(i==37)
    offs=((pitch<<1)+pitch)+1;
  if(i==38)
    offs=((pitch<<1)+pitch);
  if(i==39)
    offs=((pitch<<1)+pitch)-1;
  if(i==40)
    offs=((pitch<<1)+pitch)-2;
  if(i==41)
    offs=((pitch<<1)+pitch)-3;
  if(i==42)
    offs=(pitch<<1)-3;
  if(i==43)
    offs=pitch-3;
  if(i==44)
    offs=-3;
  if(i==45)
    offs=-pitch-3;
  if(i==46)
    offs=-(pitch<<1)-3;
  if(i==47)
    offs=-((pitch<<1)+pitch)-3;
#if (DETECTOR_BASE_NMS_SIZE >= 9)
  /* 
   * 9x9 (n=4)
   * 79 48 49 50 51 52 53 54 55
   * 78 47 24 25 26 27 28 29 56 
   * 77 46 23 8  9  10 11 30 57
   * 76 45 22 7  0  1  12 31 58
   * 75 44 21 6  x  2  13 32 59
   * 74 43 20 5  4  3  14 33 60
   * 73 42 19 18 17 16 15 34 61
   * 72 41 40 39 38 37 36 35 62
   * 71 70 69 68 67 66 65 64 63
   */
  if(i==48)
    offs=-(pitch<<2)-3;
  if(i==49)
    offs=-(pitch<<2)-2;
  if(i==50)
    offs=-(pitch<<2)-1;
  if(i==51)
    offs=-(pitch<<2);
  if(i==52)
    offs=-(pitch<<2)+1;
  if(i==53)
    offs=-(pitch<<2)+2;
  if(i==54)
    offs=-(pitch<<2)+3;
  if(i==55)
    offs=-(pitch<<2)+4;
  if(i==56)
    offs=-((pitch<<1)+pitch)+4;
  if(i==57)
    offs=-(pitch<<1)+4;
  if(i==58)
    offs=-pitch+4;
  if(i==59)
    offs=4;
  if(i==60)
    offs=pitch+4;
  if(i==61)
    offs=(pitch<<1)+4;
  if(i==62)
    offs=((pitch<<1)+pitch)+4;
  if(i==63)
    offs=(pitch<<2)+4;
  if(i==64)
    offs=(pitch<<2)+3;
  if(i==65)
    offs=(pitch<<2)+2;
  if(i==66)
    offs=(pitch<<2)+1;
  if(i==67)
    offs=(pitch<<2);
  if(i==68)
    offs=(pitch<<2)-1;
  if(i==69)
    offs=(pitch<<2)-2;
  if(i==70)
    offs=(pitch<<2)-3;
  if(i==71)
    offs=(pitch<<2)-4;
  if(i==72)
    offs=((pitch<<1)+pitch)-4;
  if(i==73)
    offs=(pitch<<1)-4;
  if(i==74)
    offs=pitch-4;
  if(i==75)
    offs=-4;
  if(i==76)
    offs=-pitch-4;
  if(i==77)
    offs=-(pitch<<1)-4;
  if(i==78)
    offs=-((pitch<<1)+pitch)-4;
  if(i==79)
    offs=-(pitch<<2)-4;
#endif /* (DETECTOR_BASE_NMS_SIZE >= 9) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 7) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 5) */
  return offs;
}
#endif /* USE_PRECALCULATED_INDICES */

__global__ void detector_base_gpu_regular_nms_kernel(const int image_width_m_borders,
                                                     const int image_height_m_borders,
                                                     const int horizontal_border,
                                                     const int vertical_border,
                                                     const int response_pitch_elements,
                                                     float * __restrict__ d_response) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // thread id X
  const int y = blockDim.y*blockIdx.y + threadIdx.y; // thread id Y
  if(x < image_width_m_borders && y < image_height_m_borders) {
    float * d_response_ptr = d_response + (y+vertical_border)*response_pitch_elements + x + horizontal_border;
    const float center_value = d_response_ptr[0];

#if USE_PRECALCULATED_INDICES == 0
    int x=0;
    int y=-1;
    int dx=1;
    int dy=0;
    bool next=false;
#endif /* USE_PRECALCULATED_INDICES */

    // Spiral generation
    #pragma unroll
    for(int i=0;i<(DETECTOR_BASE_NMS_SIZE*DETECTOR_BASE_NMS_SIZE-1);++i) {
#if USE_PRECALCULATED_INDICES
      const int j = nms_offset(i,response_pitch_elements);
#else
      const int j = y*response_pitch_elements + x;
#endif /* USE_PRECALCULATED_INDICES */

      if(d_response_ptr[j] > center_value) {
        d_response_ptr[0] = 0.0f;
        break;
      }

#if USE_PRECALCULATED_INDICES == 0
      if((x>0 && x==y) || (x==-y) || next) {
        int tmp = dx;
        dx = -dy;
        dy = tmp;
        next = false;
      } else if(x<0 && x==y) {
        next = true;
      }
      x=x+dx;
      y=y+dy;
#endif /* USE_PRECALCULATED_INDICES */
    }
  }
}

__host__ void detector_base_gpu_regular_nms(const int image_width,
                                            const int image_height,
                                            const int horizontal_border,
                                            const int vertical_border,
                                            const int response_pitch_elements,
                                            float * d_response,
                                            cudaStream_t stream) {
  const int image_width_m_borders = image_width-2*horizontal_border;
  const int image_height_m_borders = image_height-2*vertical_border;
  // Note: I'd like to have 128 threads / thread block
  const int horizontal_threads = ((image_width_m_borders%32) == 0)?32:16;
  const int vertical_threads = 128 / horizontal_threads;
  kernel_params_t p_nms = cuda_gen_kernel_params_2d(image_width_m_borders,
                                                    image_height_m_borders,
                                                    horizontal_threads,
                                                    vertical_threads);
  detector_base_gpu_regular_nms_kernel<<<p_nms.blocks_per_grid,p_nms.threads_per_block,0,stream>>>(
                                     image_width_m_borders,
                                     image_width_m_borders,
                                     horizontal_border,
                                     vertical_border,
                                     response_pitch_elements,
                                     d_response);
  CUDA_KERNEL_CHECK();
}

template<bool strictly_greater>
__global__ void detector_base_gpu_grid_nms_kernel(const int level,
                                                  const int min_level,
                                                  const int image_width,
                                                  const int image_height,
                                                  const int horizontal_border,
                                                  const int vertical_border,
                                                  const int cell_size_width,
                                                  const int cell_size_height,
                                                  const int response_pitch_elements,
                                                  const float * __restrict__ d_response,
                                                  float2 * __restrict__ d_pos,
                                                  float * __restrict__ d_score,
                                                  int * __restrict__ d_level) {
  // Various identifiers
  const int x = cell_size_width * blockIdx.x + threadIdx.x;
  const int y = cell_size_height * blockIdx.y + threadIdx.y;
  const int cell_id = gridDim.x * blockIdx.y + blockIdx.x;
  const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
  const int lane_id = thread_id & 0x1F;
  const int warp_id = thread_id >> 5;
  const int warp_cnt = (blockDim.x * blockDim.y + 31) >> 5;
  // Selected maximum response
  float max_x = static_cast<float>(x);
  float max_y = 0.f;
  float max_resp = 0.0f;

  if(x < image_width && y < image_height) {
    if(threadIdx.x == 0 && threadIdx.y == 0) {
      // the very first thread in the threadblock, initializes the cell score
      if(level == min_level) {
        d_score[cell_id] = 0.0f;
      }
    }

    /*
     * Note to future self:
     * basically, we perform NMS on every line in the cell just like a regular NMS, BUT
     * we go in a spiral and check if any of the neigbouring values is higher than our current one.
     * If it is higher, than we set our current value to zero.
     * We DO NOT write to the response buffer, we keep everything in registers. Also,
     * we do not use if-else for checking values, we use my signbit trick. This latter is amazing, because
     * it completely avoids warp divergence.
     * Then once all threads in a warp figured out whether the value they are looking at was supressed
     * or not, they reduce with warp-level intrinsics to lane 0.
     */

    // Maximum value
    // Location: x,y -> x is always threadIdx.x
    int max_y_tmp = 0;

    // border remains the same irrespective of pyramid level
    int image_width_m_border = image_width-horizontal_border;
    int image_height_m_border = image_height-vertical_border;
    if(x >= horizontal_border && x < image_width_m_border) {
      // we want as few idle threads as possible, hence we shift them according to y
      // note: we shift down all lines within the block
      int cell_top_to_border = vertical_border-(cell_size_height*blockIdx.y);
      int y_offset = cell_top_to_border>0?cell_top_to_border:0;
      int gy = y + y_offset;
      int box_line = threadIdx.y + y_offset;
      const float * d_response_ptr = d_response + gy*response_pitch_elements + x;
      int response_offset = blockDim.y*response_pitch_elements;
      for(; (box_line < cell_size_height) && (gy < image_height_m_border); box_line += blockDim.y
        , d_response_ptr += response_offset
        , gy += blockDim.y) {
        // acquire the center response value
        float center_value = d_response_ptr[0];

        // Perform spiral NMS
#if (USE_PRECALCULATED_INDICES == 0)
        int x=0;
        int y=-1;
        int dx=1;
        int dy=0;
        bool next=false;
#endif /* USE_PRECALCULATED_INDICES */

        #pragma unroll
        for(int i=0;i<(DETECTOR_BASE_NMS_SIZE*DETECTOR_BASE_NMS_SIZE-1);++i) {
#if USE_PRECALCULATED_INDICES
          int j = nms_offset(i,response_pitch_elements);
#else
          int j = y*response_pitch_elements+x;
#endif /* USE_PRECALCULATED_INDICES */

          // Perform non-maximum suppression
          if(strictly_greater) {
            center_value *= -0.5f*(-1.0f+copysignf(1.0f,d_response_ptr[j]-center_value));
          } else {
            center_value *= 0.5f*(1.0f+copysignf(1.0f,center_value-d_response_ptr[j]));
          }
          /*
           * Note to future self:
           * Interestingly on Maxwell (960M), checking for equivalence with 0.0f, results
           * in better runtimes.
           * However, on Pascal (Tegra X2) this increases the runtime, hence we opted for
           * not checking it for now.
           */
#if 0
          if(center_value==0.0f) break; // we should check it on the Jetson
#endif /* 0 */

#if (USE_PRECALCULATED_INDICES == 0)
          if((x>0 && x==y) || (x==-y) || next) {
            // if we reached the limit, change how we proceed
            // essentially we change the direction
            int tmp = dx;
            dx = -dy;
            dy = tmp;
            next = false;
          } else if(x<0 && x==y) {
            next = true;
          }
          x = x+dx;
          y = y+dy;
#endif /* USE_PRECALCULATED_INDICES */
        }

        // NMS is over, is this value greater than the previous one?
        if(center_value > max_resp) {
           max_resp = center_value;
           max_y_tmp = gy;
        }
      }
    }
    // Perform conversion
    max_y = static_cast<float>(max_y_tmp);
  }

  // Reduce the maximum location to thread 0 within each warp
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    float max_resp_new = __shfl_down_sync(WARP_MASK, max_resp, offset);
    float max_x_new    = __shfl_down_sync(WARP_MASK, max_x, offset);
    float max_y_new    = __shfl_down_sync(WARP_MASK, max_y, offset);
    if(max_resp_new > max_resp) {
      max_resp = max_resp_new;
      max_x = max_x_new;
      max_y = max_y_new;
    }
  }

  // now each warp's lane 0 has the maximum value of its cell
  // reduce in shared memory
  // each warp's lane 0 writes to shm
  // resp, x, y, (level - not used)
  extern __shared__ float s[];
  float * s_data = s + (warp_id << 2);
  float scale = static_cast<float>(1<<level);

  if(lane_id == 0) {
    s_data[0] = max_resp;
    s_data[1] = max_x;
    s_data[2] = max_y;
  }
  __syncthreads();
  // threadId x & y 0 reduces the warp results
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    s_data = s + 4; // skip self results
    for(int i=1;i<warp_cnt;i++,s_data += 4) {
      float max_resp_s = s_data[0];
      float max_x_s    = s_data[1];
      float max_y_s    = s_data[2];
      if(max_resp_s > max_resp) {
        max_resp = max_resp_s;
        max_x = max_x_s;
        max_y = max_y_s;
      }
    }

    if(d_score[cell_id] < max_resp) {
      d_score[cell_id] = max_resp;
      d_pos[cell_id].x = max_x * scale;
      d_pos[cell_id].y = max_y * scale;
      d_level[cell_id] = level;
    }
  }
}

__host__ void detector_base_gpu_grid_nms(const int image_level,
                                         const int min_image_level,
                                         const int image_width,
                                         const int image_height,
                                         const int horizontal_border,
                                         const int vertical_border,
                                         const int cell_size_width,
                                         const int cell_size_height,
                                         const int horizontal_cell_num,
                                         const int vertical_cell_num,
                                         const bool strictly_greater,
                                         /* pitch in bytes/sizeof(float) */
                                         const int response_pitch_elements,
                                         const float * d_response,
                                         float2 * d_pos,
                                         float * d_score,
                                         int * d_level,
                                         cudaStream_t stream) {
  const int cell_size_width_level  = (cell_size_width  >> image_level);
  const int cell_size_height_level = (cell_size_height >> image_level);
  int target_threads_per_block = 128;
  kernel_params_t p;
  p.threads_per_block.x = cell_size_width_level;
  p.threads_per_block.y = max(1,min(target_threads_per_block/cell_size_width_level,cell_size_height_level));
  p.threads_per_block.z = 1;
  p.blocks_per_grid.x = horizontal_cell_num;
  p.blocks_per_grid.y = vertical_cell_num;
  p.blocks_per_grid.z = 1;
  // shared memory allocation
  int launched_warp_count = (p.threads_per_block.x*p.threads_per_block.y*p.threads_per_block.z+32-1)/32;
  std::size_t shm_mem_size = launched_warp_count*4*sizeof(float);

  decltype(&detector_base_gpu_grid_nms_kernel<false>) kernel;
  if(strictly_greater) {
    kernel = detector_base_gpu_grid_nms_kernel<true>;
  } else {
    kernel = detector_base_gpu_grid_nms_kernel<false>;
  }
  kernel<<<p.blocks_per_grid,p.threads_per_block,shm_mem_size,stream>>>(
                                                image_level,
                                                min_image_level,
                                                image_width,
                                                image_height,
                                                horizontal_border,
                                                vertical_border,
                                                cell_size_width_level,
                                                cell_size_height_level,
                                                response_pitch_elements,
                                                d_response,
                                                d_pos,
                                                d_score,
                                                d_level);
  CUDA_KERNEL_CHECK();
}

} // namespace vilib
