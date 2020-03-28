/*
 * Tests for the FAST feature detector functionalities
 * test_fast.h
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
#include <unordered_map>
#include "test/test_base.h"
#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

class TestFAST : public TestBase {
public:
  TestFAST(const char * file_path, const int max_image_num = -1);
  ~TestFAST(void);
protected:
  bool run(void);
  bool run_benchmark(std::vector<vilib::Statistics> & stat_cpu,
                     std::vector<vilib::Statistics> & stat_gpu);
  void display_features_additive(const cv::Mat & level0,
                                 const std::unordered_map<int,int> & points_combined,
                                 bool draw_cells);

  // Instantiated detectors
  std::shared_ptr<vilib::DetectorBase> detector_cpu_;
  std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
};
