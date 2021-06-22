/*
 * Copyright (c) 2019-2021 Philipp Foehn,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <ros/ros.h>

#include <memory>

#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/feature_tracker/feature_tracker_gpu.h"

namespace vilib {

struct DetectorOptions {
  int cell_width{32};
  int cell_height{32};
  int min_level{0};
  int max_level{2};
  int horizontal_border{8};
  int vertical_border{8};
};

struct HarrisOptions : public DetectorOptions {
  HarrisOptions() = default;
  HarrisOptions(const DetectorOptions &other) : DetectorOptions(other) {}
  conv_filter_border_type border_type{conv_filter_border_type::BORDER_SKIP};
  float harris_k{0.04};
  float quality_level{0.1};
};

struct ShiTomasiOptions : public DetectorOptions {
  ShiTomasiOptions() = default;
  ShiTomasiOptions(const DetectorOptions &other) : DetectorOptions(other) {}
  conv_filter_border_type border_type{conv_filter_border_type::BORDER_SKIP};
  float quality_level{0.1};
};

struct FastOptions : public DetectorOptions {
  FastOptions() = default;
  FastOptions(const DetectorOptions &other) : DetectorOptions(other) {}
  float epsilon{10.0};
  int arc_length{10};
  fast_score score{SUM_OF_ABS_DIFF_ON_ARC};
};

struct VilibParams {
  VilibParams(const ros::NodeHandle &pnh);
  VilibParams(const VilibParams &other) = delete;
  ~VilibParams() = default;

  bool load(const ros::NodeHandle &pnh);
  bool valid();

  static bool toBorderType(const std::string &name,
                           conv_filter_border_type *const type);
  static bool toFastScoreType(const std::string &name, fast_score *const type);

  std::shared_ptr<FeatureTrackerGPU> createFeatureTracker() const;
  std::shared_ptr<DetectorBaseGPU> createFeatureDetector() const;

  size_t numberOfPyramidLevels() const { return n_pyramid_levels_; }


  int image_width{640};
  int image_height{480};
  size_t n_pyramid_levels_{0};
  int cell_width{32};
  int cell_height{32};

  int n_cameras{1};

  bool publish_debug_image{false};

  FeatureTrackerOptions feature_tracker_options;
  std::unique_ptr<HarrisOptions> harris_options;
  std::unique_ptr<ShiTomasiOptions> shitomasi_options;
  std::unique_ptr<FastOptions> fast_options;
};

}  // namespace vilib
