/*
 * Tests for the FAST feature detector
 * test_fast.cpp
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

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include "test/feature_detection/test_fast.h"
#include "vilib/preprocess/pyramid.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/feature_detection/fast/rosten/fast_cpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/config.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

using namespace vilib;

// Frame preprocessing
#define PYRAMID_LEVELS                       1
#define PYRAMID_MIN_LEVEL                    0
#define PYRAMID_MAX_LEVEL                    PYRAMID_LEVELS

// FAST detector parameters
#define FAST_EPSILON                         (10.0f)
#define FAST_MIN_ARC_LENGTH                  10
// Remark: the Rosten CPU version only works with 
//         SUM_OF_ABS_DIFF_ON_ARC and MAX_THRESHOLD
#define FAST_SCORE                           SUM_OF_ABS_DIFF_ON_ARC

// NMS parameters
#define HORIZONTAL_BORDER                    0
#define VERTICAL_BORDER                      0
#define CELL_SIZE_WIDTH                      32
#define CELL_SIZE_HEIGHT                     32

// Test framework options
#define DISPLAY_PYRAMID_CPU                  0
#define DISPLAY_DETECTED_FEATURES_CPU        0
#define DISPLAY_DETECTED_FEATURES_GPU        0
#define ENABLE_CPU_VERSION                   1
#define ENABLE_GPU_VERSION                   1
// Remark: the subset verification only works with the scores mentioned above
//         for the CPU version
#define ENABLE_SUBSET_VERIFICATION           1
#define ENABLE_SUBSET_VERIFICATION_MSG       1
#define ENABLE_SUBSET_VERIFICATION_IMG       0
#define ENABLE_SUBSET_VERIFICATION_IMG_SAVE  0

// Test framework statistics
#define STAT_ID_DETECTOR_TIMER               0
#define STAT_ID_FEATURE_COUNT                1

TestFAST::TestFAST(const char * file_path, const int max_image_num) :
  TestBase("FAST detector",file_path,max_image_num) {
}

TestFAST::~TestFAST(void) {
}

bool TestFAST::run(void) {
  // Create the detector statistics
  std::vector<Statistics> stat_cpu, stat_gpu;
  stat_cpu.emplace_back("[usec]","FAST");
  stat_cpu.emplace_back("[1]","FAST feature count");
  stat_gpu.emplace_back("[usec]","FAST");
  stat_gpu.emplace_back("[1]","FAST feature count");

  // Create detectors
  if(!load_image_dimensions()) {
    // Could not acquire the initialization parameters
    return false;
  }
  // CPU
#if ENABLE_CPU_VERSION
  detector_cpu_.reset(new rosten::FASTCPU<false>(image_width_,
                                          image_height_,
                                          CELL_SIZE_WIDTH,
                                          CELL_SIZE_HEIGHT,
                                          PYRAMID_MIN_LEVEL,
                                          PYRAMID_MAX_LEVEL,
                                          HORIZONTAL_BORDER,
                                          VERTICAL_BORDER,
                                          FAST_EPSILON,
                                          FAST_MIN_ARC_LENGTH,
                                          FAST_SCORE));
#endif /* ENABLE_CPU_VERSION */
  // GPU
#if ENABLE_GPU_VERSION
  detector_gpu_.reset(new FASTGPU(image_width_,
                                  image_height_,
                                  CELL_SIZE_WIDTH,
                                  CELL_SIZE_HEIGHT,
                                  PYRAMID_MIN_LEVEL,
                                  PYRAMID_MAX_LEVEL,
                                  HORIZONTAL_BORDER,
                                  VERTICAL_BORDER,
                                  FAST_EPSILON,
                                  FAST_MIN_ARC_LENGTH,
                                  FAST_SCORE));
#endif /* ENABLE_GPU_VERSION */

  // Initialize the pyramid pool
  PyramidPool::init(1,
                    image_width_,
                    image_height_,
                    1,  // grayscale
                    PYRAMID_LEVELS,
                    IMAGE_PYRAMID_MEMORY_TYPE);

  // Run benchmark suite (it will call run_benchmark()) for us
  bool success = run_benchmark_suite(stat_cpu,stat_gpu);

  // Deinitialize the pyramid pool (for consecutive tests)
  PyramidPool::deinit();

#if ENABLE_SUBSET_VERIFICATION == 0
  std::cout << " Note: No verification performed" << std::endl;
#endif /* ENABLE_SUBSET_VERIFICATION == 0 */

  return success;
}

bool TestFAST::run_benchmark(std::vector<vilib::Statistics> & stat_cpu,
                             std::vector<vilib::Statistics> & stat_gpu) {
  Timer timer;
  std::vector<cv::Mat> image_pyramid;
  // CPU -----------------------------------------------------------------------
#if ENABLE_CPU_VERSION
  timer.start();
  pyramid_create_cpu(image_,image_pyramid,PYRAMID_LEVELS,false);
  // Reset detector's grid
  detector_cpu_->reset();
  // Do the detection
  detector_cpu_->detect(image_pyramid);
  timer.stop();
  // Add statistics
  stat_cpu[STAT_ID_DETECTOR_TIMER].add(timer.elapsed_usec());
  stat_cpu[STAT_ID_FEATURE_COUNT].add(detector_cpu_->count());
  // Display results
#if DISPLAY_PYRAMID_CPU
  pyramid_display(image_pyramid);
#endif /* DISPLAY_PYRAMID_CPU */
#if DISPLAY_DETECTED_FEATURES_CPU
  detector_cpu_->displayFeatures("FAST detector (CPU)",image_pyramid,true);
#endif /* DISPLAY_DETECTED_FEATURES_CPU */
#else
  (void)stat_cpu;
  pyramid_create_cpu(image_,image_pyramid,PYRAMID_LEVELS,false);
#endif /* ENABLE_CPU_VERSION */

  // GPU -----------------------------------------------------------------------
#if ENABLE_GPU_VERSION
  timer.start();
  // Create a Frame (image upload, pyramid)
  std::shared_ptr<Frame> frame0(new Frame(image_,0,PYRAMID_LEVELS));
  // Reset detector's grid
  // Note: this step could be actually avoided with custom processing
  detector_gpu_->reset();
  // Do the detection
  detector_gpu_->detect(frame0->pyramid_);
  timer.stop();
  // Add statistics
  stat_gpu[STAT_ID_DETECTOR_TIMER].add(timer.elapsed_usec());
  stat_gpu[STAT_ID_FEATURE_COUNT].add(detector_gpu_->count());
  // Display results
#if DISPLAY_DETECTED_FEATURES_GPU
  detector_gpu_->displayFeatures("FAST detector (GPU)",image_pyramid,true,true);
#endif /* DISPLAY_DETECTED_FEATURES_GPU */
#else
  (void)stat_gpu;
#endif /* ENABLE_GPU_VERSION */

#if ENABLE_SUBSET_VERIFICATION && ENABLE_CPU_VERSION && ENABLE_GPU_VERSION
  /*
   * Note to future self:
   * verify, that the output of the GPU feature detector is the
   * subset of the CPU feature detector
   */
  auto & points_cpu = detector_cpu_->getPoints();
  auto & points_gpu = detector_gpu_->getPoints();
  auto & points_gpu_grid = detector_gpu_->getGrid();

  // O(N+M) verification
  bool success = true;
  std::size_t points_missing = 0;
  std::unordered_map<int, int> points_combined;
  points_combined.reserve(points_cpu.size());
  for(auto it=points_cpu.begin(); it != points_cpu.end(); ++it) {
    int key = ((int)it->x_) | (((int)it->y_)<<16);
    points_combined.emplace(key,1);
  }
  for(std::size_t i=0;i<points_gpu.size();++i) {
    if(!points_gpu_grid.isOccupied(i)) continue;
    int key = ((int)points_gpu[i].x_) | (((int)points_gpu[i].y_)<<16);
    if(points_combined.find(key) != points_combined.end()) {
      // found
      points_combined[key]=3;
    } else {
      // not found
      points_combined[key]=2;
#if ENABLE_SUBSET_VERIFICATION_MSG
      std::cout << " Point missing: (x=" << points_gpu[i].x_ << ", y=" 
                                         << points_gpu[i].y_ << ", s="
                                         << points_gpu[i].score_  << ", l="
                                         << points_gpu[i].level_ << ")" << std::endl;
#endif /* ENABLE_SUBSET_VERIFICATION_MSG */
      ++points_missing;
    }
  }
#if ENABLE_SUBSET_VERIFICATION_MSG
  if(points_missing > 0) {
    std::cout << " Total missing point count: " << points_missing << std::endl;
  }
#endif /* ENABLE_SUBSET_VERIFICATION_MSG */
#if ENABLE_SUBSET_VERIFICATION_IMG
  display_features_additive(image_pyramid[0],points_combined,true);
#endif /* ENABLE_SUBSET_VERIFICATION_IMG */
  return success;
#else
  return true;
#endif /* ENABLE_SUBSET_VERIFICATION */
}

void TestFAST::display_features_additive(const cv::Mat & level0,
                                         const std::unordered_map<int,int> & points_combined,
                                         const bool draw_cells) {
  cv::Mat canvas;
  cv::cvtColor(level0,canvas,cv::COLOR_GRAY2RGB);

  if(draw_cells) {
    std::size_t n_rows = (level0.rows + CELL_SIZE_HEIGHT-1)/CELL_SIZE_HEIGHT;
    std::size_t n_cols = (level0.cols + CELL_SIZE_WIDTH -1)/CELL_SIZE_WIDTH;
    for(std::size_t r=0;r<n_rows;++r) {
        for(std::size_t c=0;c<n_cols;++c) {
          cv::rectangle(canvas,
                        cv::Point(c*CELL_SIZE_WIDTH,r*CELL_SIZE_HEIGHT),
                        cv::Point((c+1)*CELL_SIZE_WIDTH,(r+1)*CELL_SIZE_HEIGHT),
                        cv::Scalar(244,215,66), // B,G,R
                        1,
                        8,
                        0);
        }
      }
  }

  // draw circles for the identified keypoints
  for(auto it=points_combined.begin(); it != points_combined.end(); ++it) {
    int x = (it->first & 0xFFFF) * 1024;
    int y = ((it->first >> 16) & 0xFFFF) * 1024;

    cv::Scalar color; // B,G,R
    int thickness = 1;
    if(it->second == 3) {
      color = cv::Scalar(0,255,255);
    } else if(it->second == 2) {
      color = cv::Scalar(255,0,0);
      thickness = 3;
    } else if(it->second == 1) {
      color = cv::Scalar(0,0,255);
    }
    cv::circle(canvas,
                cv::Point(x,y),
                1*3*1024,
                color,
                thickness,
                8,
                10);
  }
  cv::imshow("Feature detection comparison", canvas);
#if ENABLE_SUBSET_VERIFICATION_IMG_SAVE
  cv::imwrite("lenna_compare_" STRINGIFY(FAST_SCORE) ".png",canvas);
#endif /* ENABLE_SUBSET_VERIFICATION_IMG_SAVE */
  cv::waitKey();
}