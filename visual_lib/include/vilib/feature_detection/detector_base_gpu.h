/*
 * Base class for GPU feature detectors
 * detector_base_gpu.h
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

#pragma once

#include <vector>
#include <functional>
#include "vilib/feature_detection/config.h"
#include "vilib/feature_detection/detector_base.h"
#include "vilib/common/types.h"

namespace vilib {

class DetectorBaseGPU : public DetectorBase {
public:
  DetectorBaseGPU(const std::size_t image_width,
                  const std::size_t image_height,
                  const std::size_t cell_size_width,
                  const std::size_t cell_size_height,
                  const std::size_t min_level,
                  const std::size_t max_level,
                  const std::size_t horizontal_border,
                  const std::size_t vertical_border,
                  const bool subpixel_refinement,
                  const bool replace_on_same_level_only);
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
  // Feature grid variables
  // Host pointers
  float * h_pos_;
  float * h_score_;
  int * h_level_;
  // Device pointers
  float2 * d_pos_;
  int * d_level_;
  float * d_score_;
  bool subpixel_refinement_;
  bool replace_on_same_level_only_;
  // Stream to use
  cudaStream_t stream_;
};

} // namespace vilib
