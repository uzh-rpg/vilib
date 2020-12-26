/*
 * Tests for convolutional filters
 * test_conv_filter.cpp
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
#include "vilib/preprocess/conv_filter.h"
#include "vilib/preprocess/conv_filter_col.h"
#include "vilib/preprocess/conv_filter_row.h"
#include "vilib/timer.h"
#include "vilib/timergpu.h"
#include "test/preprocess/test_conv_filter.h"
#include "test/common.h"

using namespace vilib;

#define ENABLE_2D_SEPARATED                     1
#define ENABLE_1D_ROW                           1
#define ENABLE_1D_COLUMN                        1

TestConvFilter::TestConvFilter(const char * image_path) :
  TestBase("Convolutional Filters",image_path) {
}

TestConvFilter::~TestConvFilter() {
}

bool TestConvFilter::test_2d_separated(void) {
  std::cout << "2D separated" << std::endl;
  conv_filter_type_t filter_type     = conv_filter_type::GAUSSIAN_FILTER_3X3;
  conv_filter_type_t filter_type_row = conv_filter_type::GAUSSIAN_FILTER_1X3;
  conv_filter_type_t filter_type_col = conv_filter_type::GAUSSIAN_FILTER_1X3;
  bool success = true;
  for(int i=0;i<6;++i) { 
    conv_filter_border_type_t border_type = static_cast<conv_filter_border_type>(i);
    std::cout << " Border type: " << border_type << std::endl;

    // Run the convolution on the CPU
    Timer cpu_time("CPU serial");
    cpu_time.start();
    conv_filter_cpu(image_.data,
                    image_.step,
                    convfilter_image_cpu_.data,
                    convfilter_image_cpu_.step,
                    image_width_,
                    image_height_,
                    filter_type,
                    border_type);
    cpu_time.stop();
    cpu_time.display_usec();

    // Run the separated convolution on the GPU
    Timer gpu_time("GPU CUDA");
    TimerGPU gpu_kernel_time("GPU CUDA (only kernel)");
    gpu_time.start();
    cudaMemcpy(d_image_in_,image_.data,image_size_,cudaMemcpyHostToDevice);
    gpu_kernel_time.start();
    conv_filter_sep_gpu<unsigned char, float, unsigned char>(
                      d_image_in_,
                      image_width_,
                      (float*)d_image_tmp_,
                      image_width_,
                      d_image_out_,
                      image_width_,
                      image_width_,
                      image_height_,
                      filter_type_row,
                      filter_type_col,
                      border_type);
    gpu_kernel_time.stop();
    cudaMemcpy(convfilter_image_gpu_.data,d_image_out_,image_size_,cudaMemcpyDeviceToHost);
    gpu_time.stop();
    gpu_time.display_usec();
    gpu_kernel_time.sync();
    gpu_kernel_time.display_usec();

    int skip_pxl = border_type == conv_filter_border_type::BORDER_SKIP ? 1 : 0;
    success = this->compare_images(convfilter_image_cpu_,
                                        convfilter_image_gpu_,
                                        1,true,true,skip_pxl,skip_pxl,skip_pxl,skip_pxl) && success;
  }
  std::cout << "Success: " << (success?TEXT_OK:TEXT_FAIL) << std::endl;
  return success;
}

bool TestConvFilter::test_1d_row(void) {
  std::cout << "1D row" << std::endl;
  conv_filter_type_t filter_type = conv_filter_type::DIFFERENCE_FILTER_1X3;
  bool skip_first_and_last_row = false;
  bool success = true;
  for(int i=0;i<6;++i) {
    conv_filter_border_type_t border_type = static_cast<conv_filter_border_type_t>(i);
    std::cout << " Border type: " << border_type << std::endl;
    
    // Run the convolution on the CPU
    Timer cpu_time("CPU serial");
    cpu_time.start();
    conv_filter_row_cpu(image_.data,
                        image_.step,
                        convfilter_image_cpu_.data,
                        convfilter_image_cpu_.step,
                        image_width_,
                        image_height_,
                        filter_type,
                        border_type,
                        skip_first_and_last_row);
    cpu_time.stop();
    cpu_time.display_usec();

    Timer gpu_time("GPU CUDA");
    TimerGPU gpu_kernel_time("GPU CUDA (only kernel)");
    gpu_time.start();
    cudaMemcpy(d_image_in_,image_.data,image_size_,cudaMemcpyHostToDevice);
    gpu_kernel_time.start();
    conv_filter_row_gpu<unsigned char, unsigned char>(
                        d_image_in_,
                        image_width_,
                        d_image_out_,
                        image_width_,
                        image_width_,
                        image_height_,
                        filter_type,
                        border_type,
                        skip_first_and_last_row);
    gpu_kernel_time.stop();
    cudaMemcpy(convfilter_image_gpu_.data,d_image_out_,image_size_,cudaMemcpyDeviceToHost);
    gpu_time.stop();
    gpu_time.display_usec();
    gpu_kernel_time.sync();
    gpu_kernel_time.display_usec();

    int skip_pixel_top_bottom = (skip_first_and_last_row ? 1 : 0);
    int skip_pixel_left_right = (border_type == conv_filter_border_type::BORDER_SKIP ? 1 : 0);
    success = this->compare_images(convfilter_image_cpu_,
                                  convfilter_image_gpu_,
                                  1,true,true,
                                  skip_pixel_top_bottom,skip_pixel_left_right,skip_pixel_top_bottom,skip_pixel_left_right) && success;
  }
  std::cout << "Success: " << (success?TEXT_OK:TEXT_FAIL) << std::endl;
  return success;
}

bool TestConvFilter::test_1d_column(void) {
  std::cout << "1D column" << std::endl;
  conv_filter_type_t filter_type = conv_filter_type::DIFFERENCE_FILTER_1X3;
  bool skip_first_and_last_col = false;
  bool success = true;
  for(int i=0;i<6;++i) {
    conv_filter_border_type_t border_type = static_cast<conv_filter_border_type>(i);
    std::cout << " Border type: " << border_type << std::endl;

    // Run the convolution on the CPU
    Timer cpu_time("CPU serial");
    cpu_time.start();
    conv_filter_col_cpu(image_.data,
                        image_.step,
                        convfilter_image_cpu_.data,
                        convfilter_image_cpu_.step,
                        image_width_,
                        image_height_,
                        filter_type,
                        border_type,
                        skip_first_and_last_col);
    cpu_time.stop();
    cpu_time.display_usec();

    Timer gpu_time("GPU CUDA");
    TimerGPU gpu_kernel_time("GPU CUDA (only kernel)");
    gpu_time.start();
    cudaMemcpy(d_image_in_,image_.data,image_size_,cudaMemcpyHostToDevice);
    gpu_kernel_time.start();
    conv_filter_col_gpu<unsigned char, unsigned char>(
                        d_image_in_,
                        image_width_,
                        d_image_out_,
                        image_width_,
                        image_width_,
                        image_height_,
                        filter_type,
                        border_type,
                        skip_first_and_last_col);
    gpu_kernel_time.stop();
    cudaMemcpy(convfilter_image_gpu_.data,d_image_out_,image_size_,cudaMemcpyDeviceToHost);
    gpu_time.stop();
    gpu_time.display_usec();
    gpu_kernel_time.sync();
    gpu_kernel_time.display_usec();

    int skip_pixel_top_bottom = (border_type == conv_filter_border_type::BORDER_SKIP ? 1 : 0);
    int skip_pixel_left_right = (skip_first_and_last_col?1:0);
    success = this->compare_images(convfilter_image_cpu_,
                                convfilter_image_gpu_,
                                1,true,true,
                                skip_pixel_top_bottom,skip_pixel_left_right,skip_pixel_top_bottom,skip_pixel_left_right) && success;
  }
  std::cout << "Success: " << (success?TEXT_OK:TEXT_FAIL) << std::endl;
  return success;
}

bool TestConvFilter::run(void) {
  // This test suite only supports standalone images
  if(is_list_) return false;
  
  // Load image
  load_image(cv::IMREAD_GRAYSCALE,false,false);

  convfilter_image_cpu_ = cv::Mat(image_height_,image_width_,CV_8UC1);
  convfilter_image_gpu_ = cv::Mat(image_height_,image_width_,CV_8UC1);

  cudaMalloc((void**)&d_image_in_, image_size_);
  cudaMalloc((void**)&d_image_tmp_, image_size_*4);
  cudaMalloc((void**)&d_image_out_, image_size_);

  // Run separate tests
  bool separated_2d = ENABLE_2D_SEPARATED?test_2d_separated():true;
  bool row_1d       = ENABLE_1D_ROW      ?test_1d_row()      :true;
  bool column_1d    = ENABLE_1D_COLUMN   ?test_1d_column()   :true;

  cudaFree(d_image_in_);
  cudaFree(d_image_tmp_);
  cudaFree(d_image_out_);

  std::cout << "Overall" << std::endl;

  return (separated_2d && row_1d && column_1d);
}