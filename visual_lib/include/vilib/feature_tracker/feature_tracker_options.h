/*
 * Feature tracker Options
 * feature_tracker_options.h
 */

// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <vector>
#include "vilib/feature_tracker/config.h"

namespace vilib {

struct FeatureTrackerOptions {
  /// We do the Lucas Kanade tracking in a pyramidal way. max_level specifies the
  /// coarsest pyramidal level to optimize. For an image resolution of (640x480)
  /// we set this variable to 4 if you have an image with double the resolution,
  /// increase this number by one.
  int klt_max_level = 4;

  /// Similar to klt_max_level, this is the coarsest level to search for.
  /// if you have a really high resolution image and you don't extract
  /// features down to the lowest level you can set this number larger than 0.
  int klt_min_level = 0;

  /// Patch-size to use on each pyramid level.
  /// Note: the patch size on all levels need to conform to
  ///       32 % patch_size[level] == 0
  std::vector<int> klt_patch_sizes = {16, 16, 16, 8, 8};

  /// KLT maximum iteration count per level
  /// NOTE: this parameter is a precompiler macro, so that the corresponding loop
  ///       can be unrolled
  int klt_max_iter = FEATURE_TRACKER_MAX_ITERATION_COUNT;

  /// KLT termination criterion.
  double klt_min_update_squared = 0.0005;

  /// If number of tracks falls below this threshold, detect new features.
  size_t min_tracks_to_detect_new_features = 100;

  /// Reset tracker before detecting new features. This means that all active
  /// tracks are always the same age, and the reference frame is the same
  /// for all feature tracks. This can be good for VO initialization.
  bool reset_before_detection = true;

  /// Use only the best N features during detection (according to score)
  /// Note: setting -1 will use all the detected features in the grid cells
  ///       that do not contain a tracked feature yet
  int use_best_n_features = -1;

  /// Use the first observation as klt template. If set to false, then the
  /// last observation is used, which results in more feature drift.
  /// Note: if the klt template is the first observation, the reference patch
  ///       and the Hessian's inverse only needs to be computed once
  bool klt_template_is_first_observation = true;

  /// Add illumination compensation
  /// Note: only applicable to the GPU version
  bool affine_est_offset = false;
  bool affine_est_gain = false;
};

} // namespace vilib
