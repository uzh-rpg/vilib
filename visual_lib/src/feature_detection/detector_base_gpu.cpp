/*
 * Base class for GPU feature detectors
 * detector_base_gpu.cpp
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

#include <opencv2/highgui.hpp>
#include "vilib/cuda_common.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/feature_detection/detector_base_gpu_cuda_tools.h"
#include "vilib/feature_detection/detector_benchmark.h"

namespace vilib {

DetectorBaseGPU::DetectorBaseGPU(const std::size_t image_width,
                                 const std::size_t image_height,
                                 const std::size_t cell_size_width,
                                 const std::size_t cell_size_height,
                                 const std::size_t min_level,
                                 const std::size_t max_level,
                                 const std::size_t horizontal_border,
                                 const std::size_t vertical_border,
                                 const bool strictly_greater) :
  DetectorBase(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               horizontal_border,
               vertical_border),
  strictly_greater_(strictly_greater),
  stream_(0) {
  assert((cell_size_width%32) == 0);
  assert((cell_size_height%static_cast<std::size_t>(pow(2,max_level-1))) == 0);
  /*
   * Preallocate response buffers (atm 1 for each level)
   *
   * Note to future self and others:
   * because we don't allocate space for anything below min_level_,
   * if min_level_ is not zero, then all the consecutive levels will be shifted
   * in the responses_ vector.
   */
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t pitch_bytes;
    float * data;
    std::size_t width = (image_width/(1<<l));
    std::size_t height = (image_height/(1<<l));
    CUDA_API_CALL(cudaMallocPitch((void**)&data,
                                  &pitch_bytes,
                                  width*sizeof(float),
                                  height));
    struct FeatureResponse response(width,height,pitch_bytes,data);
    responses_.push_back(response);
  }
  /*
   * Preallocate FeaturePoint struct of arrays
   * Note to future self:
   * we use SoA, because of the efficient bearing vector calculation
   * float x_
   * float y_:                                      | 2x float
   * float score_:                                  | 1x float
   * int level_:                                    | 1x int
   */
  const std::size_t nr_4bytes = 0 + 4;
  const std::size_t bytes_per_featurepoint = sizeof(float)*nr_4bytes;
  feature_cell_count_ = n_cols_ * n_rows_;
  feature_grid_bytes_ = feature_cell_count_*bytes_per_featurepoint;
  CUDA_API_CALL(cudaMalloc((void**)&d_feature_grid_,feature_grid_bytes_));
  CUDA_API_CALL(cudaMallocHost((void**)&h_feature_grid_,feature_grid_bytes_));
  // Initialize pointers
  // Device & Host pointers
  h_pos_ = h_feature_grid_;
  d_pos_ = (float2*)(d_feature_grid_ );
  h_score_ = (h_feature_grid_ + feature_cell_count_*2);
  d_score_ = (d_feature_grid_ + feature_cell_count_*2);
  h_level_ = (int*)(h_feature_grid_ + feature_cell_count_*3);
  d_level_ = (int*)(d_feature_grid_ + feature_cell_count_*3);
}

void DetectorBaseGPU::setStream(cudaStream_t stream) {
  stream_ = stream;
}

cudaStream_t DetectorBaseGPU::getStream(void) {
  return stream_;
}

DetectorBaseGPU::~DetectorBaseGPU(void) {
  // Free response buffers
  for(std::size_t l=0;l<responses_.size();++l) {
    CUDA_API_CALL(cudaFree(responses_[l].data_));
  }
  // Free grid buffers
  CUDA_API_CALL(cudaFree(d_feature_grid_));
  CUDA_API_CALL(cudaFreeHost(h_feature_grid_));
}

void DetectorBaseGPU::copyGridToHost(void) {
  CUDA_API_CALL(cudaMemcpyAsync(h_feature_grid_,
                                d_feature_grid_,
                                feature_grid_bytes_,
                                cudaMemcpyDeviceToHost,
                                stream_));
  CUDA_API_CALL(cudaStreamSynchronize(stream_));
}

void DetectorBaseGPU::copyResponseTo(const std::size_t level, cv::Mat & response) const {
  assert(level < responses_.size());
  assert(level >= min_level_);
  assert(level < max_level_);
  std::size_t level_resp = level-min_level_;
  CUDA_API_CALL(cudaMemcpy2DAsync(response.data,
                                  response.step,
                                  responses_[level_resp].data_,
                                  responses_[level_resp].pitch_bytes_,
                                  response.cols*sizeof(float),
                                  response.rows,
                                  cudaMemcpyDeviceToHost,
                                  stream_));
  CUDA_API_CALL(cudaStreamSynchronize(stream_));
}

void DetectorBaseGPU::processResponse(void) {
  // Perform GRID NMS on the responses
  BENCHMARK_START_DEVICE(DetectorBenchmark,NMS,stream_);
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t l_resp = l-min_level_;
#if 1
    detector_base_gpu_grid_nms(l,
                               min_level_,
                               responses_[l_resp].width_,
                               responses_[l_resp].height_,
                               horizontal_border_,
                               vertical_border_,
                               cell_size_width_,
                               cell_size_height_,
                               n_cols_,
                               n_rows_,
                               strictly_greater_,
                               /* wpitch in bytes/sizeof(float) */
                               responses_[l_resp].pitch_elements_,
                               responses_[l_resp].data_,
                               d_pos_,
                               d_score_,
                               d_level_,
                               stream_);
#else
    // Just for comparison purposes in the paper
    detector_base_gpu_regular_nms(responses_[l_resp].width_,
                                  responses_[l_resp].height_,
                                  horizontal_border_,
                                  vertical_border_,
                                  /* pitch in bytes/sizeof(float) */
                                  responses_[l_resp].pitch_elements_,
                                  responses_[l_resp].data_,
                                  stream_);
#endif /* 0 */
  }
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,NMS,stream_);
}

void DetectorBaseGPU::saveResponses(const char * prefix) const {
  if(stream_ == NULL) {
    CUDA_API_CALL(cudaDeviceSynchronize());
  } else {
    CUDA_API_CALL(cudaStreamSynchronize(stream_));
  }
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t l_resp = l-min_level_;
    cv::Mat response = cv::Mat(responses_[l_resp].height_,responses_[l_resp].width_,CV_32FC1);
    copyResponseTo(l,response);
    std::string filename = prefix;
    filename += "_l";
    filename += std::to_string(l);
    filename += ".tiff";
    // save response image
    cv::imwrite(filename.c_str(),response);
  }
}

void DetectorBaseGPU::processGrid(void) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  for(std::size_t i=0;i<feature_cell_count_;++i) {
    if(h_score_[i] > 0.0f) {
        keypoints_[i] = FeaturePoint((double)h_pos_[2*i],
                                     (double)h_pos_[2*i+1],
                                     (double)h_score_[i],
                                     (int)h_level_[i]);
      grid_.setOccupied(i);
    }
  }
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

void DetectorBaseGPU::processGridAndThreshold(float quality_level) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  // Find the maximum score in the grid
  // Note: this could be found also on the GPU side easily
  const float max_element = *(std::max_element(h_score_,h_score_+feature_cell_count_));
  const float threshold = max_element*quality_level;
  for(std::size_t i=0;i<feature_cell_count_;++i) {
    if(h_score_[i] > threshold) {
      keypoints_[i] = FeaturePoint((double)h_pos_[2*i],
                                    (double)h_pos_[2*i+1],
                                    (double)h_score_[i],
                                    (int)h_level_[i]);
      grid_.setOccupied(i);
    }
  }
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

void DetectorBaseGPU::processGridCustom(std::function<void(const std::size_t &,
                                                           const float *,
                                                           const float *,
                                                           const int *)> callback) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  callback(feature_cell_count_,h_pos_,h_score_,h_level_);
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

} // namespace vilib
