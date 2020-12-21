/*
 * Base class for GPU feature detectors
 * detector_base_gpu.h
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

#pragma once

#include <vector>
#include <functional>
#include "vilib/feature_detection/config.h"
#include "vilib/feature_detection/detector_base.h"
#include "vilib/common/types.h"

namespace vilib {

class DetectorBaseGPU : public DetectorBase<true> {
public:
  DetectorBaseGPU(const std::size_t image_width,
                  const std::size_t image_height,
                  const std::size_t cell_size_width,
                  const std::size_t cell_size_height,
                  const std::size_t min_level,
                  const std::size_t max_level,
                  const std::size_t horizontal_border,
                  const std::size_t vertical_border,
                  // A response at (x,y) must be strictly greater than its neighborhood
                  // otherwise it is suppressed
                  const bool strictly_greater);
  virtual ~DetectorBaseGPU(void);

  void setStream(cudaStream_t stream);
  cudaStream_t getStream(void);

  virtual void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid) = 0;
  virtual void detect(const std::vector<std::shared_ptr<Subframe>> & pyramid,
                      std::function<void(const std::size_t & /* cell count */,
                                         const float *       /* pos */,
                                         const float *       /* score */,
                                         const int *         /* level */)> callback) = 0;
protected:
  struct FeatureResponse {
    const std::size_t width_;
    const std::size_t height_;
    const std::size_t pitch_elements_;
    const std::size_t pitch_bytes_;
    float * data_;

    FeatureResponse(const std::size_t width,
                    const std::size_t height,
                    const std::size_t & pitch_bytes,
                    float * data):
      width_(width),
      height_(height),
      pitch_elements_(pitch_bytes/sizeof(float)),
      pitch_bytes_(pitch_bytes),
      data_(data) {}
  };

  void copyResponseTo(const std::size_t level, cv::Mat & response) const;
  void saveResponses(const char * prefix) const;
  void copyGridToHost(void);
  void processResponse(void);
  void processGrid(void);
  void processGridAndThreshold(float quality_level);
  void processGridCustom(std::function<void(const std::size_t & /* cell count */,
                                            const float *       /* pos */,
                                            const float *       /* score */,
                                            const int *         /* level */)> callback);

  std::vector<struct FeatureResponse> responses_;
  // Feature grid
  std::size_t feature_cell_count_;
  std::size_t feature_grid_bytes_;
  float * d_feature_grid_;
  float * h_feature_grid_;
  // NMS parameters
  bool strictly_greater_;
  // Feature grid variables
  // Host pointers
  float * h_pos_;
  float * h_score_;
  int * h_level_;
  // Device pointers
  float2 * d_pos_;
  int * d_level_;
  float * d_score_;
  // Stream to use
  cudaStream_t stream_;
};

} // namespace vilib
