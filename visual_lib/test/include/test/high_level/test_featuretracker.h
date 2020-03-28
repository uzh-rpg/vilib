/*
 * Tests for feature tracking functionalities
 * test_featuretracker.h
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
