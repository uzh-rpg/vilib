/*
 * Tests for image pyramid functionalities
 * test_pyramid.cpp
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

#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include "vilib/preprocess/pyramid.h"
#include "vilib/timer.h"
#include "vilib/timergpu.h"
#include "test/preprocess/test_pyramid.h"
#include "test/common.h"

using namespace vilib;

#define PYRAMID_LEVELS                       5
#define DISPLAY_OUTPUT_IMAGES                0
#define REPETITION_COUNT                     100

TestPyramid::TestPyramid(const char * image_path) :
  TestBase("Image Pyramid",image_path) {
}

TestPyramid::~TestPyramid() {
}

bool TestPyramid::run(void) {
  // This test suite only supports standalone images
  if(is_list_) return false;
  
  // Load image
  load_image(cv::IMREAD_GRAYSCALE,false,false);

  // Run the pyramid creation on the CPU
  std::vector<cv::Mat> img_pyramid_cpu;
  Timer img_pyramid_cpu_time("CPU (w/o. preallocated array)");
  for(std::size_t r=0;r<REPETITION_COUNT;++r) {
    if(img_pyramid_cpu.size()) {
      img_pyramid_cpu.clear();
    }
    img_pyramid_cpu_time.start();
    pyramid_create_cpu(image_,
                      img_pyramid_cpu,
                      PYRAMID_LEVELS,
                      true);
    img_pyramid_cpu_time.stop();
    img_pyramid_cpu_time.add_to_stat_n_reset();
  }

  // Run the pyramid creation on the GPU
  std::vector<unsigned char *> d_img_pyramid_gpu;
  std::vector<cv::Mat> h_img_pyramid_gpu;
  std::vector<std::size_t> img_pyramid_gpu_width;
  std::vector<std::size_t> img_pyramid_gpu_height;
  std::vector<std::size_t> img_pyramid_gpu_pitch;

  preallocate_pyramid_gpu(d_img_pyramid_gpu,
                          img_pyramid_gpu_width,
                          img_pyramid_gpu_height,
                          img_pyramid_gpu_pitch,
                          PYRAMID_LEVELS);
  preallocate_pyramid_cpu(h_img_pyramid_gpu,
                          PYRAMID_LEVELS);
  cudaMemcpy2D(d_img_pyramid_gpu[0],
               img_pyramid_gpu_pitch[0],
               image_.data,
               image_width_,
               image_width_,
               image_height_,
               cudaMemcpyHostToDevice);
  Timer img_pyramid_gpu_cpu_time("GPU Host (w. preallocated array)");
  TimerGPU img_pyramid_gpu_time("GPU Device (w. preallocated array)");
  for(std::size_t r = 0;r<REPETITION_COUNT;++r) {
    img_pyramid_gpu_time.start();
    img_pyramid_gpu_cpu_time.start();
    pyramid_create_gpu(d_img_pyramid_gpu,
                       img_pyramid_gpu_width,
                       img_pyramid_gpu_height,
                       img_pyramid_gpu_pitch,
                       PYRAMID_LEVELS,
                       0);
    img_pyramid_gpu_cpu_time.stop();
    img_pyramid_gpu_time.stop();
    img_pyramid_gpu_time.add_to_stat_n_reset();
    img_pyramid_gpu_cpu_time.add_to_stat_n_reset();
  }
  copy_pyramid_from_gpu(d_img_pyramid_gpu,
                        h_img_pyramid_gpu,
                        img_pyramid_gpu_pitch,
                        PYRAMID_LEVELS);
  deallocate_pyramid_gpu(d_img_pyramid_gpu);
#if DISPLAY_OUTPUT_IMAGES
  pyramid_display(h_img_pyramid_gpu);
#endif /* DISPLAY_OUTPUT_IMAGES */

  // Display statistics
  img_pyramid_cpu_time.display_stat_usec();
  img_pyramid_gpu_time.display_stat_usec();
  img_pyramid_gpu_cpu_time.display_stat_usec();

  // Perform the verification
  return compare_image_pyramid(img_pyramid_cpu,
                               h_img_pyramid_gpu,
                               0);
}

void TestPyramid::preallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image,
                                          std::vector<std::size_t> & pyramid_width,
                                          std::vector<std::size_t> & pyramid_height,
                                          std::vector<std::size_t> & pyramid_pitch,
                                          std::size_t pyramid_levels) {
  pyramid_image.reserve(pyramid_levels);
  pyramid_width.reserve(pyramid_levels);
  pyramid_height.reserve(pyramid_levels);
  pyramid_pitch.reserve(pyramid_levels);
  for(std::size_t l=0;l<pyramid_levels;++l) {
    // allocate GPU memory
    unsigned char * d_image;
    std::size_t d_image_width = image_width_ >> l;
    std::size_t d_image_height = image_height_ >> l;
    std::size_t d_image_pitch;
    cudaMallocPitch((void**)&d_image,&d_image_pitch,image_width_,image_height_);
    pyramid_image.push_back(d_image);
    pyramid_width.push_back(d_image_width);
    pyramid_height.push_back(d_image_height);
    pyramid_pitch.push_back(d_image_pitch);
  }
}

void TestPyramid::deallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image) {
  for(std::size_t l=0;l<pyramid_image.size();++l) {
    cudaFree(pyramid_image[l]);
  }
}

void TestPyramid::preallocate_pyramid_cpu(std::vector<cv::Mat> & pyramid_image_cpu,
                                          std::size_t pyramid_levels) {
  pyramid_image_cpu.reserve(pyramid_levels);
  for(std::size_t l=0;l<pyramid_levels;++l) {
    std::size_t h_image_width = image_width_ >> l;
    std::size_t h_image_height = image_height_ >> l;
    pyramid_image_cpu.emplace_back(cv::Mat(h_image_height,h_image_width,CV_8U));
  }
}

void TestPyramid::copy_pyramid_from_gpu(std::vector<unsigned char*> & pyramid_image_gpu,
                                        std::vector<cv::Mat> & pyramid_image_cpu,
                                        std::vector<std::size_t> & pyramid_pitch_gpu,
                                        std::size_t pyramid_levels) {
  for(std::size_t l=0;l<pyramid_levels;++l) {
    cudaMemcpy2D(pyramid_image_cpu[l].data,
                 pyramid_image_cpu[l].cols,
                 pyramid_image_gpu[l],
                 pyramid_pitch_gpu[l],
                 pyramid_image_cpu[l].cols,
                 pyramid_image_cpu[l].rows,
                 cudaMemcpyDeviceToHost);
  }
}
