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

#include "vilib_tracker/vilib_params.hpp"

namespace vilib {

VilibParams::VilibParams(const ros::NodeHandle &pnh) { load(pnh); }

std::shared_ptr<FeatureTrackerGPU> VilibParams::createFeatureTracker() const {
  return std::make_shared<FeatureTrackerGPU>(feature_tracker_options,
                                             n_cameras);
}

std::shared_ptr<DetectorBaseGPU> VilibParams::createFeatureDetector() const {
  if (harris_options) {
    return std::make_shared<HarrisGPU>(
      image_width, image_height, harris_options->cell_width,
      harris_options->cell_height, harris_options->min_level,
      harris_options->max_level, harris_options->horizontal_border,
      harris_options->vertical_border, harris_options->border_type, true,
      harris_options->harris_k, harris_options->quality_level);
  } else if (shitomasi_options) {
    return std::make_shared<HarrisGPU>(
      image_width, image_height, shitomasi_options->cell_width,
      shitomasi_options->cell_height, shitomasi_options->min_level,
      shitomasi_options->max_level, shitomasi_options->horizontal_border,
      shitomasi_options->vertical_border, shitomasi_options->border_type, true,
      0.0, shitomasi_options->quality_level);
  } else if (fast_options) {
    return std::make_shared<FASTGPU>(
      image_width, image_height, fast_options->cell_width,
      fast_options->cell_height, fast_options->min_level,
      fast_options->max_level, fast_options->horizontal_border,
      fast_options->vertical_border, fast_options->epsilon,
      fast_options->arc_length, fast_options->score);
  } else {
    return nullptr;
  }
}

bool VilibParams::load(const ros::NodeHandle &pnh) {
  bool check = true;
  check &= pnh.getParam("image_width", image_width);
  check &= pnh.getParam("image_height", image_height);
  if (!check) ROS_ERROR("Could not load image size!");

  pnh.getParam("n_cameras", n_cameras);
  pnh.getParam("publish_debug_image", publish_debug_image);

  // Feature Tracker
  pnh.getParam("feature_tracker/reset_before_detection",
               feature_tracker_options.reset_before_detection);
  pnh.getParam("feature_tracker/use_best_n_features",
               feature_tracker_options.use_best_n_features);

  int min_tracks_to_detect_new_features;
  if (pnh.getParam("feature_tracker/min_tracks_to_detect_new_features",
                   min_tracks_to_detect_new_features))
    feature_tracker_options.min_tracks_to_detect_new_features =
      min_tracks_to_detect_new_features;

  pnh.getParam("feature_tracker/affine_est_gain",
               feature_tracker_options.affine_est_gain);
  pnh.getParam("feature_tracker/affine_est_offset",
               feature_tracker_options.affine_est_offset);

  DetectorOptions detector;
  pnh.getParam("feature_detector/cell_width", cell_width);
  pnh.getParam("feature_detector/cell_height", cell_height);
  detector.cell_width = cell_width;
  detector.cell_height = cell_height;
  pnh.getParam("feature_detector/min_level", detector.min_level);
  pnh.getParam("feature_detector/max_level", detector.max_level);
  pnh.getParam("feature_detector/horizontal_border",
               detector.horizontal_border);
  pnh.getParam("feature_detector/vertical_border", detector.vertical_border);
  n_pyramid_levels_ = 1 + detector.max_level;

  if (pnh.hasParam("feature_detector/harris")) {
    ROS_INFO("Loading Harris detector");
    HarrisOptions harris = detector;

    std::string border_type;
    pnh.getParam("feature_detector/harris/border_type", border_type);
    if (!toBorderType(border_type, &(harris.border_type)))
      ROS_WARN("Border type %s not valid!", border_type.c_str());

    pnh.getParam("feature_detector/harris/harris_k", harris.harris_k);
    pnh.getParam("feature_detector/harris/quality_level", harris.quality_level);
    harris_options = std::make_unique<HarrisOptions>(harris);
  }

  if (pnh.hasParam("feature_detector/shitomasi")) {
    ROS_INFO("Loading ShiTomasi detector");
    ShiTomasiOptions shitomasi = detector;

    std::string border_type;
    pnh.getParam("feature_detector/shitomasi/border_type", border_type);
    if (!toBorderType(border_type, &(shitomasi.border_type)))
      ROS_WARN("Border type %s not valid!", border_type.c_str());

    pnh.getParam("feature_detector/shitomasi/harris_k",
                 shitomasi.quality_level);
    shitomasi_options = std::make_unique<ShiTomasiOptions>(shitomasi);
  }

  if (pnh.hasParam("feature_detector/fast")) {
    ROS_INFO("Loading Fast detector");
    FastOptions fast = detector;
    pnh.getParam("feature_detector/fast/epsilon", fast.epsilon);
    pnh.getParam("feature_detector/fast/arc_length", fast.arc_length);
    std::string score;
    pnh.getParam("feature_detector/fast/score", score);
    fast_options = std::make_unique<FastOptions>(fast);
  }

  return valid();
}

bool VilibParams::toBorderType(const std::string &name,
                               conv_filter_border_type *const type) {
  if (name == "skip")
    *type = conv_filter_border_type::BORDER_SKIP;
  else if (name == "zero")
    *type = conv_filter_border_type::BORDER_ZERO;
  else if (name == "replicate")
    *type = conv_filter_border_type::BORDER_REPLICATE;
  else if (name == "reflect")
    *type = conv_filter_border_type::BORDER_REFLECT;
  else if (name == "wrap")
    *type = conv_filter_border_type::BORDER_WRAP;
  else if (name == "reflect_101")
    *type = conv_filter_border_type::BORDER_REFLECT_101;
  else {
    *type = conv_filter_border_type::BORDER_SKIP;
    return false;
  }
  return true;
}

bool VilibParams::toFastScoreType(const std::string &name,
                                  fast_score *const type) {
  if (name == "sum_of_abs_diff_all")
    *type = fast_score::SUM_OF_ABS_DIFF_ALL;
  else if (name == "sum_of_abs_diff_on_arc")
    *type = fast_score::SUM_OF_ABS_DIFF_ON_ARC;
  else if (name == "max_threshold")
    *type = fast_score::MAX_THRESHOLD;
  else {
    *type = fast_score::SUM_OF_ABS_DIFF_ON_ARC;
    return false;
  }
  return true;
}

bool VilibParams::valid() {
  return (bool)harris_options || (bool)shitomasi_options || (bool)fast_options;
}

}  // namespace vilib
