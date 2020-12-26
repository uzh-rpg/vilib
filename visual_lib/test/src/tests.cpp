/*
 * Test suite for the CUDA implementations
 * tests.cpp
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
#include <vector>
#include <memory>
#include "vilib/cuda_common.h"
#include "test/common.h"
#include "test/test_base.h"
#include "test/arguments.h"
// Preprocessing
#include "test/preprocess/test_pyramid.h"
#include "test/preprocess/test_conv_filter.h"
// Storage
#include "test/storage/test_subframepool.h"
#include "test/storage/test_pyramidpool.h"
// Feature detection
#include "test/feature_detection/test_fast.h"
#include "test/feature_detection/test_harris.h"
// High-level functionalities
#include "test/high_level/test_featuretracker.h"

using namespace vilib;

// Available test images
#define TEST_IMAGE_PERU_640_480                "test/images/peru_640_480.jpg"
#define TEST_IMAGE_ANGRY_BIRDS_752_480         "test/images/angry_birds_752_480.jpg"
#define TEST_IMAGE_BLENDER_HUT_752_480         "test/images/scenery/hut/00.png"
#define TEST_IMAGE_LENNA_512_512               "test/images/lenna.png"
#define TEST_IMAGE_CHESSBOARD_798_798          "test/images/chessboard_798_798.png"
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
  tests.emplace_back(new TestConvFilter(TEST_IMAGE_ANGRY_BIRDS_752_480));
  // Storage
  tests.emplace_back(new TestSubframePool());
  tests.emplace_back(new TestPyramidPool());
  // Feature detection
  tests.emplace_back(new TestFAST(TEST_IMAGE_LIST_EUROC_752_480,100));
  tests.emplace_back(new TestHarris(TEST_IMAGE_LIST_EUROC_752_480,100));
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
