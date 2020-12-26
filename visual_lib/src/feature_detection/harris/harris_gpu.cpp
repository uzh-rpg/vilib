/*
 * Harris/Shi-Tomasi feature detector on the GPU
 * harris_gpu.cpp
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

#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu_cuda_tools.h"
#include "vilib/feature_detection/detector_benchmark.h"
#include "vilib/preprocess/conv_filter.h"
#include "vilib/cuda_common.h"
#include "vilib/timer.h"

namespace vilib {

// Copy back the response images (as debug)
#define SAVE_RESPONSE                 0
// Number of memory buffers to be preallocated
#define DEVICE_BUFFER_COUNT           4
// Buffer indexes
#define BUFFER_ID_TEMP                0
#define BUFFER_ID_DX                  1
#define BUFFER_ID_DY                  2
#define BUFFER_ID_DXDY                3

// We need a minimal border, due to various reasons:
// - skip pixels due to the NMS kernel
// - skip 1 pixel if BORDER_SKIP
// - skip 1 pixel due to the neighborhood summation during the corner response calculation
#define MINIMUM_BORDER(BORDER_TYPE)  ((DETECTOR_BASE_NMS_SIZE/2) + 1 + \
                                       (BORDER_TYPE==conv_filter_border_type::BORDER_SKIP?1:0))
#define BUFFER(BUFFER_ID)            d_result_[BUFFER_ID]
#define PITCH(BUFFER_ID)             d_result_pitch_elements_[BUFFER_ID]

HarrisGPU::HarrisGPU(const std::size_t image_width,
                 const std::size_t image_height,
                 const std::size_t cell_size_width,
                 const std::size_t cell_size_height,
                 const std::size_t min_level,
                 const std::size_t max_level,
                 const std::size_t horizontal_border,
                 const std::size_t vertical_border,
                 const conv_filter_border_type_t filter_border_type,
                 const bool use_harris,
                 const float harris_k,
                 const float quality_level) :
  DetectorBaseGPU(image_width,
                  image_height,
                  cell_size_width,
                  cell_size_height,
                  min_level,
                  max_level,
                  std::max(static_cast<std::size_t>(MINIMUM_BORDER(filter_border_type)),horizontal_border),
                  std::max(static_cast<std::size_t>(MINIMUM_BORDER(filter_border_type)),vertical_border),
                  false),
  filter_border_type_(filter_border_type),
  quality_level_(quality_level),
  use_harris_(use_harris),
  harris_k_(harris_k),
  d_result_pitch_elements_(0) {
  // Allocate temporary result buffers
  // We only allocate buffer for the largest image
  for(std::size_t i=0;i<DEVICE_BUFFER_COUNT;++i) {
    float * temp;
    std::size_t result_pitch_bytes;
    CUDA_API_CALL(cudaMallocPitch((void**)&temp,&result_pitch_bytes,image_width*sizeof(float),image_height));
    d_result_.push_back(temp);
    d_result_pitch_elements_.push_back(result_pitch_bytes / sizeof(float));
  }
}

HarrisGPU::~HarrisGPU(void) {
  for(std::size_t i=0;i<DEVICE_BUFFER_COUNT;++i) {
    CUDA_API_CALL(cudaFree(d_result_[i]));
  }
}

void HarrisGPU::detectBase(const std::vector<std::shared_ptr<Subframe>> & pyramid) {
  /*
   * Harris corner detection process:
   * - the image is already in GPU memory
   * - determine corner points
   * - apply non-maximum suppression on them
   * - populate the feature grid
   * - copy the feature grid back to memory
   */
  BENCHMARK_START_DEVICE(DetectorBenchmark,CRF,stream_);
  for(std::size_t level=min_level_;level<pyramid.size() && level<max_level_;++level) {
    std::size_t level_resp = level-min_level_;
    float input_scale = 1.0f/255.f;
    // Dx
    conv_filter_sep_gpu<unsigned char,float,float>(pyramid[level]->data_,pyramid[level]->pitch_,
                          BUFFER(BUFFER_ID_TEMP),PITCH(BUFFER_ID_TEMP),
                          BUFFER(BUFFER_ID_DX),PITCH(BUFFER_ID_DX),
                          pyramid[level]->width_,
                          pyramid[level]->height_,
                          conv_filter_type::DIFFERENCE_FILTER_1X3,
                          conv_filter_type::SOBEL_FILTER_1X3,
                          filter_border_type_,
                          input_scale,
                          stream_);
    // Dy
    conv_filter_sep_gpu<unsigned char,float,float>(pyramid[level]->data_,pyramid[level]->pitch_,
                          BUFFER(BUFFER_ID_TEMP),PITCH(BUFFER_ID_TEMP),
                          BUFFER(BUFFER_ID_DY),PITCH(BUFFER_ID_DY),
                          pyramid[level]->width_,
                          pyramid[level]->height_,
                          conv_filter_type::SOBEL_FILTER_1X3,
                          conv_filter_type::DIFFERENCE_FILTER_1X3,
                          filter_border_type_,
                          input_scale,
                          stream_);
    // DxDy = Dx * Dy
    harris_gpu_array_multiply(BUFFER(BUFFER_ID_DX),
                              PITCH(BUFFER_ID_DX),
                              BUFFER(BUFFER_ID_DY),
                              PITCH(BUFFER_ID_DY),
                              BUFFER(BUFFER_ID_DXDY),
                              PITCH(BUFFER_ID_DXDY),
                              pyramid[level]->width_,
                              pyramid[level]->height_,
                              stream_);
    // Dx^2 = Dx * Dx
    harris_gpu_array_multiply(BUFFER(BUFFER_ID_DX),
                              PITCH(BUFFER_ID_DX),
                              BUFFER(BUFFER_ID_DX),
                              PITCH(BUFFER_ID_DX),
                              BUFFER(BUFFER_ID_DX),
                              PITCH(BUFFER_ID_DX),
                              pyramid[level]->width_,
                              pyramid[level]->height_,
                              stream_);
    // Dy^2 = Dy * Dy
    harris_gpu_array_multiply(BUFFER(BUFFER_ID_DY),
                              PITCH(BUFFER_ID_DY),
                              BUFFER(BUFFER_ID_DY),
                              PITCH(BUFFER_ID_DY),
                              BUFFER(BUFFER_ID_DY),
                              PITCH(BUFFER_ID_DY),
                              pyramid[level]->width_,
                              pyramid[level]->height_,
                              stream_);
    // Response score
    harris_gpu_calc_corner_response(BUFFER(BUFFER_ID_DX),
                                    PITCH(BUFFER_ID_DX),
                                    BUFFER(BUFFER_ID_DY),
                                    PITCH(BUFFER_ID_DY),
                                    BUFFER(BUFFER_ID_DXDY),
                                    PITCH(BUFFER_ID_DXDY),
                                   responses_[level_resp].data_,
                                   responses_[level_resp].pitch_elements_,
                                   pyramid[level]->width_,
                                   pyramid[level]->height_,
                                   filter_border_type_,
                                   use_harris_,
                                   harris_k_,
                                   stream_);
  }
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,CRF,stream_);
#if SAVE_RESPONSE
  saveResponses("harris_gpu");
#endif /* SAVE_RESPONSE */
  // Perform non-maximum suppression on the response images
  processResponse();
}

void HarrisGPU::detect(const std::vector<std::shared_ptr<Subframe>> & pyramid) {
  detectBase(pyramid);
  processGridAndThreshold(quality_level_);
}

void HarrisGPU::detect(const std::vector<std::shared_ptr<Subframe>> & pyramid,
                     std::function<void(const std::size_t &  /* cell count */,
                                        const float *  /* pos */,
                                        const float *  /* score */,
                                        const int *  /* level */)> callback) {
  detectBase(pyramid);
  processGridCustom(callback);
}

} // namespace vilib
