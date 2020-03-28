/*
 * Test suite for the CUDA implementations
 * tests.cpp
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

#include <iostream>
#include <vector>
#include <memory>
#ifdef CUSTOM_OPENCV_SUPPORT
#include <opencv2/cudev/common.hpp>
#endif /* CUSTOM_OPENCV_SUPPORT */
#include "vilib/cuda_common.h"
#include "test/common.h"
#include "test/test_base.h"
#include "test/arguments.h"
// Preprocessing
#include "test/preprocess/test_pyramid.h"
// Storage
#include "test/storage/test_subframepool.h"
#include "test/storage/test_pyramidpool.h"
// Feature detection
#include "test/feature_detection/test_fast.h"
// High-level functionalities
#include "test/high_level/test_featuretracker.h"

using namespace vilib;

// Available test images
#define TEST_IMAGE_PERU_640_480                "test/images/peru_640_480.jpg"
#define TEST_IMAGE_ANGRY_BIRDS_752_480         "test/images/angry_birds_752_480.jpg"
#define TEST_IMAGE_BLENDER_HUT_752_480         "test/images/scenery/hut/00.png"
#define TEST_IMAGE_LENNA_512_512               "test/images/lenna.png"
/*
 * To have the complete EUROC Machine Hall 01 dataset frames at 752x480 & 640x480:
 * cd images
 * ./create_feature_detector_evaluation_data.sh
 * <follow the instructions>
 * .. then you can use the testcases below:
 */
#define TEST_IMAGE_LIST_EUROC_752_480          "test/images/euroc/image_list_752_480.txt"
#define TEST_IMAGE_LIST_EUROC_640_480          "test/images/euroc/image_list_640_480.txt"

int main(int argc, char * argv[]) {
  // Save arguments
  init_arguments(argc,argv);

  // Initializations
#ifdef CUSTOM_OPENCV_SUPPORT
  /*
   * Placing cv::Mats in page-locked memory buffers
   * The CUDA memcpy-s became super-fast, but operations on the matrices on the CPU got somewhat slower:
   * - the copyies got fast: because we dont need to copy the area to a
   *   pinned location first, hence we save the first step of a regular H2D,D2H transfer
   * - operations on the CPU got slower, because the pinned-memory content is NOT
   *   cached on the CPU
   */
  cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));
#endif /* CUSTOM_OPENCV_SUPPORT */

  /*
   * Note to future self:
   * we explicitly initialize the CUDA runtime here, so that benchmarks later
   * are not skewed by the API's initialization.
   */
  if(!cuda_initialize()) {
    return -1;
  }

  std::vector<struct TestCase> tests;
  // Preprocessing
  tests.emplace_back(new TestPyramid(TEST_IMAGE_ANGRY_BIRDS_752_480));
  // Storage
  tests.emplace_back(new TestSubframePool());
  tests.emplace_back(new TestPyramidPool());
  // Feature detection
  tests.emplace_back(new TestFAST(TEST_IMAGE_LIST_EUROC_752_480));
  // High level
  tests.emplace_back(new TestFeatureTracker(TEST_IMAGE_LIST_EUROC_752_480,100));

  // Execute tests
  bool overall_success = true;
  for(auto it = tests.begin();it != tests.end();++it) {
    bool repeat = false;
    do {
      for(std::size_t i=0;i<it->rep_cnt_;++i) {
        overall_success = overall_success && it->test_->evaluate();
      }
      if(it->fn_) {
        repeat = it->fn_(it->test_);
      }
    } while(repeat);
  }
  std::cout << "### Overall" << std::endl;
  std::cout << "Success: " << (overall_success?TEXT_OK:TEXT_FAIL) << std::endl;

  return 0;
}
