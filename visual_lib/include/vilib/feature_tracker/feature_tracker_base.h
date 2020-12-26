/*
 * Feature tracker base class
 * feature_tracker_base.h
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

#include <memory>
#include <vector>
#include "vilib/feature_tracker/feature_tracker_options.h"
#include "vilib/common/frame.h"
#include "vilib/common/framebundle.h"
#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/common/occupancy_grid_2d.h"
#include "vilib/statistics.h"

namespace vilib {

class FeatureTrackerBase {
public:
  FeatureTrackerBase(const FeatureTrackerOptions & options,
                     const std::size_t & camera_num);
  virtual ~FeatureTrackerBase(void);

  virtual void track(const std::shared_ptr<FrameBundle> & cur_frames,
                     std::size_t & total_tracked_features_num,
                     std::size_t & total_detected_features_num) = 0;
  virtual void setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                      const std::size_t & camera_id);
  void setBestNFeatures(int n);
  void setMinTracksToDetect(int n);
#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
  void showAdditionalStat(bool include_remaining);
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
  virtual void reset(void);
  void getDisparity(const double & pivot_ratio,
                    double & total_avg_disparity);
protected:
  struct FeatureTrack {
    /*
     * Note to future self:
     * - if first frame used as template, then first_frame = template_frame && first_pos = template_pos
     * - if not first frame used as template, then template_frame, and template_pos is always updated, and template_pos = current_pos
     */
    // first: the frame was the first where the feature was extracted from
    std::shared_ptr<Frame> first_frame_;
    float first_pos_[2];
    int first_level_;
    float first_score_;
    // template: the frame used as a template in KLT
    std::shared_ptr<Frame> template_frame_;
    float template_pos_[2];
    // current: the latest tracked position of the feature
    float cur_pos_[2];
    float cur_disparity_;
    // track id
    int track_id_;
    // buffer id
    std::size_t buffer_id_;
    // track life (0=newly detected feature,1<= tracked)
    std::size_t life_;

    FeatureTrack(std::shared_ptr<Frame> first_frame,
                 const float & first_pos_x,
                 const float & first_pos_y,
                 const int & first_level,
                 const float & first_score,
                 const int & track_id,
                 const std::size_t & buffer_id = 0) :
      first_frame_(first_frame),
      first_pos_{first_pos_x,first_pos_y},
      first_level_(first_level),
      first_score_(first_score),
      template_frame_(first_frame),
      template_pos_{first_pos_x,first_pos_y},
      cur_pos_{first_pos_x,first_pos_y},
      cur_disparity_(0.0f),
      track_id_(track_id),
      buffer_id_(buffer_id),
      life_(0)
      {}
  };

  void removeTracks(const std::vector<std::size_t> & ind_to_remove,
                    const std::size_t & camera_id);
  void addFeature(const std::shared_ptr<Frame> & frame,
                  const struct FeatureTrack & track);
  void addFeature(const std::shared_ptr<Frame> & frame,
                  const int & track_id,
                  const std::size_t & camera_id);

  FeatureTrackerOptions options_;
  std::vector<std::vector<struct FeatureTrack>> tracks_;
  std::vector<std::size_t> tracked_features_num_;
  std::vector<std::size_t> detected_features_num_;
#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
  Statistics life_stat_;
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
};

} // namespace vilib
