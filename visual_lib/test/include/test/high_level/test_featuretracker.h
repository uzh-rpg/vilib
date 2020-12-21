/*
 * Tests for feature tracking functionalities
 * test_featuretracker.h
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
#include "vilib/statistics.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/feature_tracker/feature_tracker_base.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"
#include "test/test_base.h"
#include "test/groundtruth/depth_image.h"

class TestFeatureTracker : public TestBase {
public:
  TestFeatureTracker(const char * file_path = NULL,const int max_image_num = -1);
  ~TestFeatureTracker(void) = default;
  void setTrackerOptions(int use_best_n, int min_tracks_to_detect);
protected:
  bool run(void);
  bool run_blender(std::vector<vilib::Statistics> & stat_cpu,
                   std::vector<vilib::Statistics> & stat_gpu);
  bool run_benchmark(std::vector<vilib::Statistics> & stat_cpu,
                     std::vector<vilib::Statistics> & stat_gpu);

  void calculate_feature_error(
        const Eigen::MatrixXd & M,
        const DepthImage & depth_image,
        const std::shared_ptr<vilib::Frame> & frame,
        vilib::Statistics & rmse_per_feature,
        vilib::Statistics & rmse_per_frame);
  void visualize_features(
      const std::shared_ptr<vilib::Frame> & frame,
      cv::Mat & display,
      bool draw_cells);
private:
  bool initialized_;
  std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
  std::shared_ptr<vilib::FeatureTrackerBase> tracker_gpu_;
};
