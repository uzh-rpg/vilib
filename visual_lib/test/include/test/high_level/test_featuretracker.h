/*
 * Tests for feature tracking functionalities
 * test_featuretracker.h
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
