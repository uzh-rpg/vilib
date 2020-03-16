/*
 * Tests for the FAST feature detector functionalities
 * test_fast.h
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
