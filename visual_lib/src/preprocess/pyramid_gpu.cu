/*
 * Functions for creating image pyramids on the GPU
 * pyramid_gpu.cu
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

#include <opencv2/imgproc.hpp>
#include "vilib/preprocess/pyramid.h"
#include "vilib/cuda_common.h"

namespace vilib {
#define USE_TEXTURE_MEMORY  0

#define USE_TEXTURE_OBJECTS 0

#if (USE_TEXTURE_MEMORY == 1) && (USE_TEXTURE_OBJECTS == 0)
static texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> d_image_in_tex;
#endif /* USE_TEXTURE_MEMORY */

#if USE_TEXTURE_MEMORY
template <typename T, const int N>
static __global__ void image_halfsample_gpu_tex_kernel(T * __restrict__ d_image_out,
#if USE_TEXTURE_OBJECTS
                                                       cudaTextureObject_t d_image_in_tex,
#endif /* USE_TEXTURE_OBJECTS */
                                                       const unsigned int width_dst_px,
                                                       const unsigned int height_dst_px,
                                                       const unsigned int pitch_dst_px) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if((x < width_dst_px) && (y < height_dst_px)) {
    const int dst = y*pitch_dst_px/N + x; //every thread writes N bytes. the next row starts at pitch_Dst_px/N
    float src_x = N*2.0f*x + 1.0f;        //every thread reads in 2x4 bytes
    float src_y = 2.0f*y + 1.0f;
    #pragma unroll N
    for(int i=0;i<N;++i,src_x += 2.0f) {
      *(((unsigned char*)(d_image_out+dst))+i) = (unsigned char)(255.0f*
#if USE_TEXTURE_OBJECTS
      tex2D<float>
#else
      tex2D
#endif /* USE_TEXTURE_OBJECTS */
        (d_image_in_tex,src_x,src_y));
    }
  }
}
#else
template <typename T, const int N>
__global__ void image_halfsample_gpu_kernel(const uchar2 * __restrict__ d_image_in,
                                 const unsigned int pitch_src_px,
                                 T * __restrict__ d_image_out,
                                 const unsigned int width_px,
                                 const unsigned int height_px,
                                 const unsigned int pitch_dst_px) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if((x < width_px) && (y < height_px)) {
    const int dst  = y*pitch_dst_px/N + x; //every thread writes N bytes. the next row starts at pitch_dst_px/N
    int src_top    = y*pitch_src_px + x*N; //every thread reads in Nx2 bytes
    int src_bottom = y*pitch_src_px + x*N + (pitch_src_px/2);
    #pragma unroll N
    for(int i=0;i<N;++i) {
      const uchar2 t2 = d_image_in[src_top++];
      const uchar2 b2 = d_image_in[src_bottom++];
      *(((unsigned char*)(d_image_out+dst))+i) = (unsigned char)(((unsigned int)t2.x + (unsigned int)t2.y + (unsigned int)b2.x + (unsigned int)b2.y)>>2);
    }
  }
}
#endif /* USE_TEXTURE_MEMORY */

static inline __host__ void pyramid_create_level_gpu(const unsigned char * d_img_src,
                                                      unsigned char * d_img_dst,
                                                      std::size_t & img_src_pitch,
                                                      std::size_t & img_dst_pitch,
                                                      std::size_t & img_src_width,
                                                      std::size_t & img_dst_width,
                                                      std::size_t & img_src_height,
                                                      std::size_t & img_dst_height,
                                                      cudaStream_t stream) {
#if USE_TEXTURE_MEMORY
#if USE_TEXTURE_OBJECTS
  cudaTextureObject_t tex_object;
  cudaResourceDesc tex_res;
  memset(&tex_res,0,sizeof(cudaResourceDesc));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = img_src_width;
  tex_res.res.pitch2D.height = img_src_height;
  tex_res.res.pitch2D.pitchInBytes = img_src_pitch;
  tex_res.res.pitch2D.devPtr = (void*)d_img_src;
  tex_res.res.pitch2D.desc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);

  cudaTextureDesc tex_desc;
  memset(&tex_desc,0,sizeof(cudaTextureDesc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeNormalizedFloat;

  // create texture object on the fly
  CUDA_API_CALL(cudaCreateTextureObject(&tex_object,&tex_res,&tex_desc,NULL));
#else
  // Bind the texture memory
  d_image_in_tex.addressMode[0] = cudaAddressModeClamp;
  d_image_in_tex.addressMode[1] = cudaAddressModeClamp;
  d_image_in_tex.filterMode     = cudaFilterModeLinear;
  d_image_in_tex.normalized     = 0;
  d_image_in_tex.channelDesc    = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
  CUDA_API_CALL(cudaBindTexture2D(NULL,&d_image_in_tex,d_img_src,&d_image_in_tex.channelDesc,img_src_width,img_src_height,img_src_pitch));
#endif /* USE_TEXTURE_OBJECTS */
#else
  // Unused parameters
  (void)img_src_width;
  (void)img_src_height;
#endif /* USE_TEXTURE_MEMORY */

  // Use the most efficient vectorized version
  for(unsigned int v=4;v>0;v=v/2) {
    if(img_dst_width % v == 0) {
      const unsigned int img_dst_width_n = img_dst_width/v;
      const unsigned int thread_num_x = min(64,((img_dst_width_n+32-1)/32)*32);
      const kernel_params_t p = cuda_gen_kernel_params_2d(img_dst_width_n,img_dst_height,thread_num_x,2);
      switch(v) {
        case 1:
#if USE_TEXTURE_MEMORY
          image_halfsample_gpu_tex_kernel<uchar1,1><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar1*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar1,1><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar1*)d_img_dst,
            (unsigned int)img_dst_width,
            (unsigned int)img_dst_height,
            (unsigned int)img_dst_pitch
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
        case 2:
#if USE_TEXTURE_MEMORY
          image_halfsample_gpu_tex_kernel<uchar2,2><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar2,2><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar2*)d_img_dst,
            (unsigned int)(img_dst_width_n),
            (unsigned int)(img_dst_height),
            (unsigned int)(img_dst_pitch)
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
        case 4:
#if USE_TEXTURE_MEMORY
          image_halfsample_gpu_tex_kernel<uchar4,4><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar4*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar4,4><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar4*)d_img_dst,
            (unsigned int)(img_dst_width_n),
            (unsigned int)(img_dst_height),
            (unsigned int)(img_dst_pitch)
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
      }
      break;
    }
  }

#if USE_TEXTURE_MEMORY
#if USE_TEXTURE_OBJECTS
  // Destroy the created texture object
  CUDA_API_CALL(cudaDestroyTextureObject(tex_object));
#else
  // Unbind the texture memory
  CUDA_API_CALL(cudaUnbindTexture(d_image_in_tex));
#endif /* USE_TEXTURE_OBJECTS */
#endif /* USE_TEXTURE_MEMORY */
}

__host__ void pyramid_create_gpu(std::vector<unsigned char *> & d_images,
                                 std::vector<std::size_t> & width,
                                 std::vector<std::size_t> & height,
                                 std::vector<std::size_t> & pitch,
                                 unsigned int levels,
                                 cudaStream_t stream) {
  for(std::size_t l=1;l<levels;++l) {
    pyramid_create_level_gpu(d_images[l-1],
                             d_images[l],
                             pitch[l-1],
                             pitch[l],
                             width[l-1],
                             width[l],
                             height[l-1],
                             height[l],
                             stream);
  }
}

__host__ void pyramid_create_gpu(std::vector<std::shared_ptr<Subframe>> & d_subframes,
                                 cudaStream_t stream) {
  for(std::size_t l=1;l<d_subframes.size();++l) {
    pyramid_create_level_gpu(d_subframes[l-1]->data_,
                             d_subframes[l]->data_,
                             d_subframes[l-1]->pitch_,
                             d_subframes[l]->pitch_,
                             d_subframes[l-1]->width_,
                             d_subframes[l]->width_,
                             d_subframes[l-1]->height_,
                             d_subframes[l]->height_,
                             stream);
  }
}

__host__ void pyramid_display(const std::vector<std::shared_ptr<Subframe>> & subframes) {
  for(std::size_t l=0;l<subframes.size();++l) {
    subframes[l]->display();
  }
}

} // namespace vilib
