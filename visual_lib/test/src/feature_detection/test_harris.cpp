/*
 * Tests for the Harris/Shi-Tomasi feature detector
 * test_harris.cpp
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
#include "test/feature_detection/test_harris.h"
#include "vilib/preprocess/pyramid.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/feature_detection/harris/harris_cpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/config.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

using namespace vilib;

/*
 * Note to future self and others:
 * there's some difference between the CPU and the GPU implementation.
 * the reason is three-fold:
 * - the CPU (OpenCV) implementation does not provide a score for features,
 *   hence we need to generate the score ourselves (currently hardcoded to be 1.0f)
 *   During multi-level detection, if a cell is already occupied by a feature, this
 *   feature is not going to be replaced from lower pyramid levels, because they will have
 *   the same hard-coded score. This is a deficiency of cv::goodFeaturesToTrack()
 * - the CPU (OpenCV) implementation uses a minimum euclidean distance between
 *   features. Once this is satisfied, then we apply the grid cell. (if set to 0, 
 *   this is not an issue)
 * - the GPU version skips at least 1 row and 1 column of pixels at the borders which 
 *   might contain the maximum response, and since we use relative thresholding,
 *   this might introduce a (noticable) difference between the CPU and GPU version outputs
 */

// Frame preprocessing
#define PYRAMID_LEVELS                       1
#define PYRAMID_MIN_LEVEL                    0
#define PYRAMID_MAX_LEVEL                    PYRAMID_LEVELS

// Harris/Shi-Tomasi detector parameters
#define USE_HARRIS                           true
#define HARRIS_K                             0.04f
// CPU Detector parameters
#define HARRIS_QUALITY_LEVEL                 0.1f
#define SHI_TOMASI_QUALITY_LEVEL             0.1f
#define MIN_EUCLIDEAN_DISTANCE               0.0f
#define CONV_FILTER_BORDER_TYPE              conv_filter_border_type::BORDER_SKIP

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

// Test framework statistics
#define STAT_ID_DETECTOR_TIMER               0
#define STAT_ID_FEATURE_COUNT                1

#if (USE_HARRIS == true)
#define DETECTOR_NAME                        "Harris"
#else
#define DETECTOR_NAME                        "Shi-Tomasi"
#endif /* USE_HARRIS */
#define QUALITY_LEVEL                        (USE_HARRIS?HARRIS_QUALITY_LEVEL:SHI_TOMASI_QUALITY_LEVEL)

TestHarris::TestHarris(const char * file_path, const int max_image_num) :
  TestBase(DETECTOR_NAME " detector",file_path,max_image_num) {
}

TestHarris::~TestHarris(void) {
}

bool TestHarris::run(void) {
  // Create the detector statistics
  std::vector<Statistics> stat_cpu, stat_gpu;
  stat_cpu.emplace_back("[usec]",DETECTOR_NAME);
  stat_cpu.emplace_back("[1]",DETECTOR_NAME " feature count");
  stat_gpu.emplace_back("[usec]",DETECTOR_NAME);
  stat_gpu.emplace_back("[1]",DETECTOR_NAME " feature count");

  // Create detectors
  if(!load_image_dimensions()) {
    // Could not acquire the initialization parameters
    return false;
  }
  // CPU
#if ENABLE_CPU_VERSION
  detector_cpu_.reset(new opencv::HarrisCPU<true>(image_width_,
                                                 image_height_,
                                           CELL_SIZE_WIDTH,
                                           CELL_SIZE_HEIGHT,
                                           PYRAMID_MIN_LEVEL,
                                           PYRAMID_MAX_LEVEL,
                                           HORIZONTAL_BORDER,
                                           VERTICAL_BORDER,
                                           USE_HARRIS,
                                           HARRIS_K,
                                           QUALITY_LEVEL,
                                           MIN_EUCLIDEAN_DISTANCE));
#endif /* ENABLE_CPU_VERSION */
  // GPU
#if ENABLE_GPU_VERSION
  detector_gpu_.reset(new HarrisGPU(image_width_,
                                    image_height_,
                                    CELL_SIZE_WIDTH,
                                    CELL_SIZE_HEIGHT,
                                    PYRAMID_MIN_LEVEL,
                                    PYRAMID_MAX_LEVEL,
                                    HORIZONTAL_BORDER,
                                    VERTICAL_BORDER,
                                    CONV_FILTER_BORDER_TYPE,
                                    USE_HARRIS,
                                    HARRIS_K,
                                    QUALITY_LEVEL));
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
  return success;
}

bool TestHarris::run_benchmark(std::vector<vilib::Statistics> & stat_cpu,
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
  detector_cpu_->displayFeatures(DETECTOR_NAME " detector (CPU)",image_pyramid,true);
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
  detector_gpu_->displayFeatures(DETECTOR_NAME " detector (GPU)",image_pyramid,true,true);
#endif /* DISPLAY_DETECTED_FEATURES_GPU */
#else
  (void)stat_gpu;
#endif /* ENABLE_GPU_VERSION */

  return true;
}