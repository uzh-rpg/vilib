/*
 * Functions for handling OpenCV types
 * opencv.cpp 
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>
#include "vilib/storage/opencv.h"
#include "vilib/cuda_common.h"

namespace vilib {

static inline void opencv_copy_to_image_common(cv::Mat & dst_img,
                                               const unsigned char * src_img,
                                               unsigned int src_img_width,
                                               unsigned int src_img_height,
                                               unsigned int src_img_element_size,
                                               unsigned int src_img_pitch,
                                               bool async,
                                               cudaStream_t stream_num,
                                               enum cudaMemcpyKind dir) {
  // Initialize the destination image
  // TODO: currently we only support grayscale images
  assert(src_img_element_size == 1);
  if(dst_img.cols != (int)src_img_width ||
     dst_img.rows != (int)src_img_height) {
    dst_img = cv::Mat(src_img_height,src_img_width,CV_8U);
  }

  // Do the copying
  void * dst = (void*)dst_img.data;
  const void * src = (const void *)src_img;
  std::size_t dpitch = dst_img.step;
  std::size_t spitch = src_img_pitch;
  std::size_t width  = src_img_width*src_img_element_size;
  std::size_t height = src_img_height;
  if(async) {
    CUDA_API_CALL(cudaMemcpy2DAsync(dst,dpitch,
                                    src,spitch,
                                    width,height,
                                    dir,stream_num));
  } else {
    CUDA_API_CALL(cudaMemcpy2D(dst,dpitch,
                               src,spitch,
                               width,height,
                               dir));
  }
}

static inline void opencv_copy_from_image_common(const cv::Mat & src_img,
                                                 unsigned char * dst_img,
                                                 unsigned int dst_img_pitch,
                                                 bool async,
                                                 cudaStream_t stream_num,
                                                 enum cudaMemcpyKind dir) {
  //Do the copying
  const void * src = (const void *)src_img.data;
  void * dst = (void*)dst_img;
  std::size_t dpitch = dst_img_pitch;
  std::size_t spitch = src_img.step;
  std::size_t width  = src_img.cols * src_img.elemSize();
  std::size_t height = src_img.rows;
  if(async) {
    CUDA_API_CALL(cudaMemcpy2DAsync(dst,dpitch,
                                    src,spitch,
                                    width,height,
                                    dir,stream_num));
  } else {
    CUDA_API_CALL(cudaMemcpy2D(dst,dpitch,
                               src,spitch,
                               width,height,
                               dir));
  }
}

void opencv_copy_from_image_to_gpu(const cv::Mat & h_src_img,
                                   unsigned char * d_dst_img,
                                   unsigned int dst_img_pitch,
                                   bool async,
                                   cudaStream_t stream_num) {
  opencv_copy_from_image_common(h_src_img,
                                d_dst_img,
                                dst_img_pitch,
                                async,
                                stream_num,
                                cudaMemcpyHostToDevice);
}

void opencv_copy_from_image_to_host(const cv::Mat & h_src_img,
                                    unsigned char * h_dst_img,
                                    unsigned int dst_img_pitch) {
  opencv_copy_from_image_common(h_src_img,
                                h_dst_img,
                                dst_img_pitch,
                                false,
                                0,
                                cudaMemcpyHostToHost);
}

void opencv_copy_from_host_to_image(cv::Mat & h_dst_img,
                                    const unsigned char * h_src_img,
                                    unsigned int src_img_width,
                                    unsigned int src_img_height,
                                    unsigned int src_img_element_size,
                                    unsigned int src_img_pitch) {
  opencv_copy_to_image_common(h_dst_img,
                              h_src_img,
                              src_img_width,
                              src_img_height,
                              src_img_element_size,
                              src_img_pitch,
                              false,
                              0,
                              cudaMemcpyHostToHost);
}

void opencv_copy_from_gpu_to_image(cv::Mat & dst_img,
                                   const unsigned char * d_src_img,
                                   unsigned int src_img_width,
                                   unsigned int src_img_height,
                                   unsigned int src_img_element_size,
                                   unsigned int src_img_pitch,
                                   bool async,
                                   cudaStream_t stream_num) {
  opencv_copy_to_image_common(dst_img,
                              d_src_img,
                              src_img_width,
                              src_img_height,
                              src_img_element_size,
                              src_img_pitch,
                              async,
                              stream_num,
                              cudaMemcpyDeviceToHost);
}

void opencv_register_image_in_pinned_memory(const cv::Mat & h_img) {
  /*
   * Note to future self and others:
   * on ARM platforms (like the Jetson TX2) the caching behaviour of memory areas
   * cannot be changed on the fly, hence this call does not work and returns:
   * "operation not supported".
   */
  CUDA_API_CALL(cudaHostRegister(h_img.data,
                                 h_img.total()*h_img.elemSize(),
                                 cudaHostRegisterPortable));
}

void opencv_unregister_image_from_pinned_memory(const cv::Mat & h_img) {
  CUDA_API_CALL(cudaHostUnregister(h_img.data));
}

} // namespace vilib
