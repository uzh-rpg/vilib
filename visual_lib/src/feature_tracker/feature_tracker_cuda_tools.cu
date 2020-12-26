/*
 * CUDA kernels for the feature tracker
 * feature_tracker_cuda_tools.cu
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

#include "vilib/feature_tracker/feature_tracker_cuda_tools.h"
#include "vilib/feature_tracker/config.h"
#include "vilib/cuda_common.h"

namespace vilib {
namespace feature_tracker_cuda_tools {

// Warp preliminaries
#define WARP_SIZE                       32
#define WARP_MASK                       0xFFFFFFFF
// Precalculating feature patches
#define CANDIDATES_PER_BLOCK_UPDATE     3
/*
 * Note to future self:
 * - interestingly with reference patch interpolation, the tracking performance
 *   degrades. -> DISABLE
 * - Int/Float does not seem to affect the tracking performance, but the float
 *   version is slower. -> INT
 */
#define REFERENCE_PATCH_INTERPOLATION   0
#define CANDIDATES_PER_BLOCK_TRACK      2

template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__device__ __inline__ void perform_lk(const float & min_update_squared,
                                      const int & img_width,
                                      const int & img_height,
                                      const int & img_pitch,
                                      const unsigned char * __restrict__ d_in_cur_img,
                                      const int & patch_size,
                                      const int & half_patch_size,
                                      const int & patch_stride,
                                      const T * ref_patch,
                                      const float * invH,
                                      float2 & cur_px,
                                      float2 & cur_alpha_beta,
                                      bool & converged,
                                      bool & go_to_next_level) {
  converged = false;
  go_to_next_level = false;

  // Reference patch & actual image
  const int patch_area = patch_size * patch_size;
  const int x = threadIdx.x % patch_size;
  const int y = threadIdx.x / patch_size;
  const T * it_ref_start = ref_patch + (y+1)*patch_stride + (x+1);
  const int it_ref_offset = patch_stride*WARP_SIZE/patch_size;
  const unsigned char * it_start = (const unsigned char*) d_in_cur_img -
                             (img_pitch + 1)*half_patch_size + threadIdx.x +
                             (img_pitch - patch_size)*(threadIdx.x/patch_size);
  const int it_offset = img_pitch*WARP_SIZE/patch_size;
  const int pixels_per_thread = patch_area/WARP_SIZE;

  #pragma unroll
  for(int iter=0; iter<FEATURE_TRACKER_MAX_ITERATION_COUNT;++iter) {
    if(isnan(cur_px.x) || isnan(cur_px.y)) {
      break;
    }
    int u_r = floorf(cur_px.x);
    int v_r = floorf(cur_px.y);
    if(u_r < half_patch_size ||
       v_r < half_patch_size ||
       u_r >= (img_width-half_patch_size) ||
       v_r >= (img_height-half_patch_size)) {
      // don't change the state 'converged'
      go_to_next_level = true;
      break;
    }

    // compute interpolation weights
    float subpix_x = cur_px.x-u_r;
    float subpix_y = cur_px.y-v_r;
    float wTL = (1.0f-subpix_x)*(1.0f-subpix_y);
    float wTR = subpix_x * (1.0f-subpix_y);
    float wBL = (1.0f-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    float Jres[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
      Jres[i] = 0.0f;
    }

    const uint8_t * it = it_start + u_r + v_r*img_pitch;
    const T * it_ref = it_ref_start;

    // Note: every thread computes (PATCH_SIZE*PATCH_SIZE/WARP_SIZE) pixels
    #pragma unroll
    for(int i=0;i<pixels_per_thread;++i,it+=it_offset,it_ref+=it_ref_offset) {
      // Note it cannot be read as uchar2, because it would require proper alignment
      float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[img_pitch] + wBR*it[img_pitch+1];
      float res = search_pixel - (1.0f+cur_alpha_beta.x)*(*it_ref) - cur_alpha_beta.y;
      Jres[0] += res * 0.5f * (it_ref[1] - it_ref[-1]);
      Jres[1] += res * 0.5f * (it_ref[patch_stride] - it_ref[-patch_stride]);

      // If affine compensation is used,
      // set Jres with respect to affine parameters.
      if(affine_est_offset && affine_est_gain) {
        Jres[2] += res;
        Jres[3] += res*(*it_ref);
      } else if(affine_est_offset) {
        Jres[2] += res;
      } else if(affine_est_gain) {
        Jres[2] += res*(*it_ref);
      }
    }

    // Reduce it to all lanes
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      #pragma unroll
      for(int i=0;i<4;++i) {
        Jres[i] += __shfl_xor_sync(WARP_MASK, Jres[i], offset);
      }
    }

    //update = Hinv * Jres
    //broadcast the computed values in the warp from lane 0
    float update[4];
    if(affine_est_offset && affine_est_gain) {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1] + invH[2] * Jres[2] + invH[3] * Jres[3];
      update[1] = invH[1] * Jres[0] + invH[4] * Jres[1] + invH[5] * Jres[2] + invH[6] * Jres[3];
      update[2] = invH[2] * Jres[0] + invH[5] * Jres[1] + invH[7] * Jres[2] + invH[8] * Jres[3];
      update[3] = invH[3] * Jres[0] + invH[6] * Jres[1] + invH[8] * Jres[2] + invH[9] * Jres[3];
    } else if(affine_est_gain || affine_est_offset) {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1] + invH[2] * Jres[2];
      update[1] = invH[1] * Jres[0] + invH[3] * Jres[1] + invH[4] * Jres[2];
      update[2] = invH[2] * Jres[0] + invH[4] * Jres[1] + invH[5] * Jres[2];
    } else {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1];
      update[1] = invH[1] * Jres[0] + invH[2] * Jres[1];
    }

    // Updating our estimates
    // Translational displacement
    cur_px.x -= update[0];
    cur_px.y -= update[1];

    // Illumination changes
    if(affine_est_offset && affine_est_gain) {
      cur_alpha_beta.x += update[3];
      cur_alpha_beta.y += update[2];
    } else if (affine_est_offset) {
      cur_alpha_beta.y += update[2];
    } else if(affine_est_gain) {
      cur_alpha_beta.x += update[2];
    }

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared) {
      converged=true;
      break;
    }
  }
}

template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__global__ void track_features_kernel(const int candidate_num,
                                      const int min_level,
                                      const int max_level,
                                      const float min_update_squared,
                                      const image_pyramid_descriptor_t pyramid_description,
                                      const pyramid_patch_descriptor_t  pyramid_patch_sizes,
                                      const int * __restrict__ d_indir_data,
                                      const T * __restrict__ d_patch_data,
                                      const float * __restrict__ d_hessian_data,
                                      const float2 * __restrict__ d_first_px,
                                      float2 * __restrict__ d_cur_px,
                                      float2 * __restrict__ d_cur_alpha_beta,
                                      float4 * __restrict__ d_cur_f,
                                      float  * __restrict__ d_cur_disparity) {
  const int cx = blockIdx.x * CANDIDATES_PER_BLOCK_TRACK + threadIdx.y; // candidate id
  const int pyramid_levels = max_level - min_level + 1; // number of pyramid levels computed
  if(cx < candidate_num) {
    // Acquire buffer id for the candidate
    const int bx = d_indir_data[cx];

    // Initialize input and output references
    // Remark: struct size: 64 bytes
    // Tracking
    float2 & d_cur_px_bx = d_cur_px[bx<<3]; // level 0
    float2 & d_cur_alpha_beta_bx = d_cur_alpha_beta[bx<<3];
    // Bearing vector
    float4 & d_cur_f_bx = d_cur_f[bx<<2];
    // Disparity calculation
    const float2 & d_first_px_bx = d_first_px[bx<<3];
    float & d_cur_disparity_bx = d_cur_disparity[bx<<4];

    // Patch data
    const T * d_patch_data_bx = d_patch_data + pyramid_patch_sizes.max_area*pyramid_levels*bx; // points to max_level
    // Hessian data
    const float * d_hessian_data_bx = d_hessian_data + 10*pyramid_levels*bx; // points to max_level

    /*
     * Iterate through all the selected levels,
     * and refine the current patch location
     */
    bool converged = false;
    bool go_to_next_level = true;
    float2 cur_px = d_cur_px_bx;
    float2 cur_alpha_beta = d_cur_alpha_beta_bx;
    float scale = 1.0f;
    for(int level=max_level;
        (converged || go_to_next_level) && level>=min_level;
        cur_px.x *= scale,cur_px.y *= scale,--level,d_patch_data_bx+=pyramid_patch_sizes.max_area,d_hessian_data_bx+=10) {
      // scale & patch size
      scale = (float)(1<<level);
      const float inv_scale = 1.0f/scale;
      const int & patch_size = pyramid_patch_sizes.wh[level];
      const int & half_patch_size = patch_size >> 1;
      const int & patch_stride = patch_size + 2;

      // image size
      const int & d_in_img_width  = pyramid_description.desc.w[level];
      const int & d_in_img_height = pyramid_description.desc.h[level];
      const int & d_in_img_pitch  = pyramid_description.desc.p[level];

      // update the pixel positions according to the current level
      cur_px.x = cur_px.x * inv_scale;
      cur_px.y = cur_px.y * inv_scale;

      // TODO : maybe load it into shared memory later if size is small (8x8), also the inverse hessian!
      // Check if the inverse hessian was computed for the level successfully
      if(isnan(d_hessian_data_bx[0])) {
        continue;
      }

      const unsigned char * d_in_cur_img = pyramid_description.data[level];
      // do the Lukas-Kanade on the actual level, using the reference patch
      perform_lk<T, affine_est_offset, affine_est_gain>(
                                 min_update_squared,
                                 d_in_img_width,
                                 d_in_img_height,
                                 d_in_img_pitch,
                                 d_in_cur_img,
                                 patch_size,
                                 half_patch_size,
                                 patch_stride,
                                 d_patch_data_bx,
                                 d_hessian_data_bx,
                                 cur_px,
                                 cur_alpha_beta,
                                 converged,
                                 go_to_next_level);
    }

    if(threadIdx.x == 0) {
      if(converged) {
        // Point location
        d_cur_px_bx.x = cur_px.x;
        d_cur_px_bx.y = cur_px.y;
        // Alpha-beta estimation
        if(affine_est_gain) {
          d_cur_alpha_beta_bx.x = cur_alpha_beta.x;
        }
        if(affine_est_offset) {
          d_cur_alpha_beta_bx.y = cur_alpha_beta.y;
        }
        // Bearing vector
        // TODO
        (void)d_cur_f_bx;
        // Disparity
        d_cur_disparity_bx = sqrtf((cur_px.x - d_first_px_bx.x)*(cur_px.x - d_first_px_bx.x) +
                                   (cur_px.y - d_first_px_bx.y)*(cur_px.y - d_first_px_bx.y));
      } else {
        // Initialize output to be NAN
        d_cur_px_bx.x = __int_as_float(0x7fffffff);
        d_cur_px_bx.y = __int_as_float(0x7fffffff);
      }
    }
  }
}

__host__ void track_features(const bool affine_est_offset,
                             const bool affine_est_gain,
                             const int candidate_num,
                             const int min_level,
                             const int max_level,
                             const float min_update_squared,
                             const image_pyramid_descriptor_t & pyramid_description,
                             const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                             const int * d_indir_data,
                             const unsigned char * d_patch_data,
                             const float * d_hessian_data,
                             const float2 * d_in_first_px,
                             float2 * d_in_cur_px,
                             float2 * d_in_cur_alpha_beta,
                             float4 * d_in_cur_f,
                             float  * d_in_cur_disparity,
                             cudaStream_t stream) {
  // Kernel parameters
  dim3 threads_per_block;
  threads_per_block.x = WARP_SIZE;
  threads_per_block.y = CANDIDATES_PER_BLOCK_TRACK;
  threads_per_block.z = 1;
  const int blocks_per_grid = (candidate_num + CANDIDATES_PER_BLOCK_TRACK - 1) / CANDIDATES_PER_BLOCK_TRACK;
  const int shm_per_block = 0; //CANDIDATES_PER_BLOCK_TRACK*(PATCH_STRIDE*(PATCH_SIZE+2))*sizeof(REFERENCE_PATCH_TYPE);
  // Launch kernel
  if(affine_est_offset && affine_est_gain) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,true><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else if(affine_est_offset) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,false><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else if(affine_est_gain) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,true><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,false><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  }
  CUDA_KERNEL_CHECK();
}

/* Precalculate patches & precalculate inverse Hessians  --------------------- */

template<typename T>
__device__ __inline__ bool load_ref_patch(const unsigned char * __restrict__ d_in_ref_img,
                                          const float2 & ref_px,
                                          const int & img_width,
                                          const int & img_height,
                                          const int & img_pitch,
                                          const int & patch_size,
                                          const int & half_patch_size,
                                          T * __restrict__ d_out_ref_patch) {
  // Calculate top left corner
  // and verify that the patch fits in the image
  float x_tl_f = ref_px.x - (half_patch_size+1);
  float y_tl_f = ref_px.y - (half_patch_size+1);

  int x_tl_i = floorf(x_tl_f);
  int y_tl_i = floorf(y_tl_f);
  // Note: on the right side of the rectangle, there's a +1 because of the
  //       intensity interpolation
  if(x_tl_i < 0 ||
     y_tl_i < 0 ||
#if REFERENCE_PATCH_INTERPOLATION
     (x_tl_i+patch_size+2)>=img_width ||
     (y_tl_i+patch_size+2)>=img_height
#else
     (x_tl_i+patch_size+1)>=img_width ||
     (y_tl_i+patch_size+1)>=img_height
#endif /* REFERENCE_PATCH_INTERPOLATION */
    ) {
    return false;
  }

#if REFERENCE_PATCH_INTERPOLATION
  float subpix_x = x_tl_f-x_tl_i;
  float subpix_y = y_tl_f-y_tl_i;
  float wTL = (1.0f-subpix_x)*(1.0f-subpix_y);
  float wTR = subpix_x*(1.0f-subpix_y);
  float wBL = (1.0f-subpix_x)*subpix_y;
  float wBR = 1.0f - wTL - wTR - wBL;
#endif /* REFERENCE_PATCH_INTERPOLATION */

  T * patch_ptr = d_out_ref_patch + threadIdx.x;
  #pragma unroll
  for(int id = threadIdx.x; id < (patch_size+2)*(patch_size+2); id += WARP_SIZE, patch_ptr += WARP_SIZE) {
    int x_no_offs = (id % (patch_size+2));
    int y_no_offs = (id / (patch_size+2));
    int xi = x_no_offs + x_tl_i;
    int yi = y_no_offs + y_tl_i;

    const unsigned char * ptr = d_in_ref_img + yi*img_pitch + xi;
    *patch_ptr =
#if REFERENCE_PATCH_INTERPOLATION
          (T)(wTL*((float)ptr[0]) +
              wBL*((float)ptr[img_pitch]) +
              wTR*((float)ptr[1]) +
              wBR*((float)ptr[img_pitch+1]));
#else
          (T)ptr[0];
#endif /* REFERENCE_PATCH_INTERPOLATION */
  }
  return true;
}


template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__device__ __inline__ void calc_hessian(const int & img_width,
                                        const int & img_height,
                                        const int & img_pitch,
                                        const T * ref_patch,
                                        const int & patch_size,
                                        const int & half_patch_size,
                                        const int & patch_stride,
                                        float * __restrict__ d_inv_hessian) {
  /*
   * We're exploiting the fact that H is going to be symmetric !
   * - When we estimate also affine offset AND gain:
   * J = [ 0 1 2 3 ]
   * H = | 0 1 2 3 |
   *     | x 4 5 6 |
   *     | x x 7 8 |
   *     | x x x 9 |
   *
   * - When we estimate only affine offset OR gain
   * J = [ 0 1 2 ]
   * H = | 0 1 2 |
   *     | x 3 4 |
   *     | x x 5 |
   *
   * - When we don't estimate the affine offset and gain
   * J = [ 0 1 ]
   * H = | 0 1 |
   *     | x 2 |
   */
  float H[10];
  float J[4];
  #pragma unroll
  for(int i=0;i<10;++i) {
    H[i] = 0.0f;
  }

  const int patch_area = patch_size * patch_size;
  const int x = threadIdx.x % patch_size;
  const int y = threadIdx.x / patch_size;
  const T * it_ref_start = ref_patch + (y+1)*patch_stride + (x+1); // +1 due to the 1-1-1-1 borders
  const T * it_ref = it_ref_start;
  const int it_ref_offset = patch_stride * WARP_SIZE / patch_size;
  const int pixels_per_thread = patch_area/WARP_SIZE;

  #pragma unroll
  for(int i=0;i<pixels_per_thread;++i,it_ref+=it_ref_offset) {
    // Compute J
    J[0] = 0.5f * (it_ref[1] - it_ref[-1]);
    J[1] = 0.5f * (it_ref[patch_stride] - it_ref[-patch_stride]);
    // Affine parameter estimation
    if(affine_est_offset && affine_est_gain) {
      J[2] = 1.0f;
      J[3] = it_ref[0];
    } else if(affine_est_offset) {
      J[2] = 1.0f;
    } else if(affine_est_gain) {
      J[2] = it_ref[0];
    }

    // H += J*J^Transpose (using the fact that J*J^T is going to be symmetric)
    if(affine_est_offset && affine_est_gain) {
      /*
       * H: 4x4 matrix
       *    0      1     2     3
       * 0  (0x0) * (0x1) (0x2) (0x3)
       * 1  x     (1x1) (1x2) (1x3)
       * 2  x     x     (2x2) (2x3)
       * 3  x     x     x     (3x3)
       */
       H[0] += J[0]*J[0];
       H[1] += J[0]*J[1];
       H[2] += J[0]*J[2];
       H[3] += J[0]*J[3];
       H[4] += J[1]*J[1];
       H[5] += J[1]*J[2];
       H[6] += J[1]*J[3];
       H[7] += J[2]*J[2];
       H[8] += J[2]*J[3];
       H[9] += J[3]*J[3];
    } else if(affine_est_offset || affine_est_gain) {
      /*
       * H: 3x3 matrix
       *    0      1     2
       * 0  (0x0) (0x1) (0x2)
       * 1  x     (1x1) (1x2)
       * 2  x     x     (2x2)
       */
       H[0] += J[0]*J[0];
       H[1] += J[0]*J[1];
       H[2] += J[0]*J[2];
       H[3] += J[1]*J[1];
       H[4] += J[1]*J[2];
       H[5] += J[2]*J[2];
    } else {
      /*
       img_cur_width_pitch_diff* H: 2x2 matrix
       *    0      1
       * 0  (0x0) (0x1)
       * 1  x     (1x1)
       */
      H[0] += J[0]*J[0];
      H[1] += J[0]*J[1];
      H[2] += J[1]*J[1];
    }
  }

  // Reduce it down to lane 0
  #pragma unroll
  for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    #pragma unroll
    for(int i=0;i<10;++i) {
      H[i] += __shfl_down_sync(WARP_MASK, H[i], offset);
    }
  }

  // Calculate the inverse of H
  float inv_detH;
  if(threadIdx.x == 0) {
    if(affine_est_gain && affine_est_offset) {
      // Inverse of a symmetric 4x4 matrix
      inv_detH = 1.0f /
              (H[0]*H[4]*H[7]*H[9] + 2*H[0]*H[5]*H[6]*H[8] + H[1]*H[1]*H[8]*H[8]
              + 2*H[1]*H[2]*H[5]*H[9] + 2*H[1]*H[3]*H[6]*H[7]
              + 2*H[2]*H[3]*H[4]*H[8] + H[2]*H[2]*H[6]*H[6] + H[3]*H[3]*H[5]*H[5]
              - H[0]*H[4]*H[8]*H[8] - H[0]*H[5]*H[5]*H[9] - H[0]*H[6]*H[6]*H[7]
              - H[1]*H[1]*H[7]*H[9] - 2*H[1]*H[3]*H[5]*H[8] - 2*H[1]*H[2]*H[6]*H[8]
              - H[2]*H[2]*H[4]*H[9] - 2*H[2]*H[3]*H[5]*H[6] - H[3]*H[3]*H[4]*H[7]);
      d_inv_hessian[0] = (H[4]*H[7]*H[9] + 2*H[5]*H[6]*H[8] - H[4]*H[8]*H[8] - H[5]*H[5]*H[9] - H[6]*H[6]*H[7]) * inv_detH;
      d_inv_hessian[1] = (H[1]*H[8]*H[8] + H[2]*H[5]*H[9] + H[3]*H[6]*H[7] - H[1]*H[7]*H[9] - H[2]*H[6]*H[8] - H[3]*H[5]*H[8]) * inv_detH;
      d_inv_hessian[2] = (H[1]*H[5]*H[9] + H[2]*H[6]*H[6] + H[3]*H[4]*H[8] - H[1]*H[6]*H[8] - H[2]*H[4]*H[9] - H[3]*H[5]*H[6]) * inv_detH;
      d_inv_hessian[3] = (H[1]*H[6]*H[7] + H[2]*H[4]*H[8] + H[3]*H[5]*H[5] - H[1]*H[5]*H[8] - H[2]*H[6]*H[5] - H[3]*H[4]*H[7]) * inv_detH;
      d_inv_hessian[4] = (H[0]*H[7]*H[9] + 2*H[2]*H[3]*H[8] - H[0]*H[8]*H[8] - H[2]*H[2]*H[9] - H[3]*H[3]*H[7]) * inv_detH;
      d_inv_hessian[5] = (H[0]*H[6]*H[8] + H[1]*H[2]*H[9] + H[3]*H[3]*H[5] - H[0]*H[5]*H[9] - H[2]*H[3]*H[6] - H[1]*H[3]*H[8]) * inv_detH;
      d_inv_hessian[6] = (H[0]*H[5]*H[8] + H[2]*H[2]*H[6] + H[1]*H[3]*H[7] - H[0]*H[6]*H[7] - H[1]*H[2]*H[8] - H[2]*H[3]*H[5]) * inv_detH;
      d_inv_hessian[7] = (H[0]*H[4]*H[9] + 2*H[1]*H[3]*H[6] - H[0]*H[6]*H[6] - H[1]*H[1]*H[9] - H[3]*H[3]*H[4]) * inv_detH;
      d_inv_hessian[8] = (H[0]*H[5]*H[6] + H[1]*H[1]*H[8] + H[2]*H[3]*H[4] - H[0]*H[4]*H[8] - H[1]*H[2]*H[6] - H[1]*H[3]*H[5]) * inv_detH;
      d_inv_hessian[9] = (H[0]*H[4]*H[7] + 2*H[1]*H[2]*H[5] - H[0]*H[5]*H[5] - H[1]*H[1]*H[7] - H[2]*H[2]*H[4]) * inv_detH;
    } else if(affine_est_gain || affine_est_offset) {
      // Inverse of a symmetric 3x3 matrix
      inv_detH = 1.0f / (H[0]*H[3]*H[5] + 2*H[1]*H[4]*H[2] - H[0]*H[4]*H[4] - H[2]*H[3]*H[2] - H[1]*H[1]*H[5]);
      d_inv_hessian[0] = (H[3]*H[5] - H[4]*H[4]) * inv_detH;
      d_inv_hessian[1] = (H[2]*H[4] - H[1]*H[5]) * inv_detH;
      d_inv_hessian[2] = (H[1]*H[4] - H[2]*H[3]) * inv_detH;
      d_inv_hessian[3] = (H[0]*H[5] - H[2]*H[2]) * inv_detH;
      d_inv_hessian[4] = (H[1]*H[2] - H[0]*H[4]) * inv_detH;
      d_inv_hessian[5] = (H[0]*H[3] - H[1]*H[1]) * inv_detH;
    } else {
      // Inverse of a symmetric 2x2 matrix
      inv_detH = 1.0f / (H[0]*H[2] - H[1]*H[1]);
      d_inv_hessian[0] = H[2] * inv_detH;
      d_inv_hessian[1] = -1.0f * H[1] * inv_detH;
      d_inv_hessian[2] = H[0] * inv_detH;
    }
  }
}

template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__global__ void update_tracks_kernel(const int candidate_num,
                                     const int min_level,
                                     const int max_level,
                                     const image_pyramid_descriptor_t pyramid_description,
                                     const pyramid_patch_descriptor_t pyramid_patch_sizes,
                                     const int * __restrict__ d_indir_data,
                                     const float2 * __restrict__ d_in_ref_px,
                                     T * __restrict__ d_patch_data,
                                     float * __restrict__ d_hessian_data) {
  const int cx = blockIdx.x * CANDIDATES_PER_BLOCK_UPDATE + threadIdx.y; // candidate id
  const int pyramid_levels = max_level - min_level + 1; // number of pyramid levels computed
  if(cx < candidate_num) {
    // Get the buffer id
    const int bx = d_indir_data[cx];
    // Metadata array size: 64 bytes
    const float2 & ref_px_bx = d_in_ref_px[bx<<3];
    // Patch data
    T * d_patch_data_bx = d_patch_data + pyramid_patch_sizes.max_area*pyramid_levels*bx;
    // Hessian data
    float * d_hessian_data_bx = d_hessian_data + 10*pyramid_levels*bx;

    // For all the levels, precompute the interpolated patch, and the resulting inverse Hessian
    for(int level=max_level;
        level>=min_level;
        --level,d_patch_data_bx+=pyramid_patch_sizes.max_area,d_hessian_data_bx+=10) {
      const float scale = (float)(1<<level);
      const float inv_scale = 1.0f/scale;
      const float2 ref_px_scaled = {.x = ref_px_bx.x * inv_scale, .y = ref_px_bx.y * inv_scale};

      const unsigned char * img_ref = pyramid_description.data[level];
      const int img_width  = pyramid_description.desc.w[level];
      const int img_height = pyramid_description.desc.h[level];
      const int img_pitch  = pyramid_description.desc.p[level];

      const int patch_size = pyramid_patch_sizes.wh[level];
      const int half_patch_size = patch_size >> 1;
      const int patch_stride = patch_size + 2;

      // Create the reference patch if possible with borders
      if(load_ref_patch<T>(img_ref,
                           ref_px_scaled,
                           img_width,
                           img_height,
                           img_pitch,
                           patch_size,
                           half_patch_size,
                           d_patch_data_bx) == false) {
          // To notify the subsequent kernel of this behaviour,
          // use the 0th index Hessian
          d_hessian_data_bx[0] = __int_as_float(0x7fffffff);
          continue;
      }
      __syncwarp();

      // Calculate the Hessian and its inverse
      calc_hessian<T,affine_est_offset,affine_est_gain>(img_width,
                                                        img_height,
                                                        img_pitch,
                                                        d_patch_data_bx,
                                                        patch_size,
                                                        half_patch_size,
                                                        patch_stride,
                                                        d_hessian_data_bx);
    }
  }
}

__host__ void update_tracks(const int candidate_num,
                            const bool affine_est_offset,
                            const bool affine_est_gain,
                            const int min_level,
                            const int max_level,
                            const image_pyramid_descriptor_t & pyramid_description,
                            const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                            const int * d_indir_data,
                            const float2 * d_in_ref_px,
                            unsigned char * d_patch_data,
                            float * d_hessian_data,
                            cudaStream_t stream) {
  dim3 threads_per_block;
  threads_per_block.x = WARP_SIZE;
  threads_per_block.y = CANDIDATES_PER_BLOCK_UPDATE;
  threads_per_block.z = 1;
  const int blocks_per_grid = (candidate_num + CANDIDATES_PER_BLOCK_UPDATE - 1) / CANDIDATES_PER_BLOCK_UPDATE;
  // Launch kernel
  if(affine_est_offset && affine_est_gain) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,true><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else if(affine_est_offset) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,false><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else if(affine_est_gain) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,true><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,false><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  }
  CUDA_KERNEL_CHECK();
}

} // namespace feature_tracker_cuda_tools
} // namespace vilib
