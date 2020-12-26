/*
 * Base class for testing various functionalities
 * test_base.cpp
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
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include "test/test_base.h"
#include "test/common.h"
#include "test/arguments.h"
#include "vilib/benchmark.h"
#include "vilib/feature_detection/detector_benchmark.h"

using namespace vilib;

TestBase::TestBase(const char * name, const char * file_path, const int max_image_num):
  name_(name),
  file_path_(file_path),
  max_image_num_(max_image_num),
  success_(true),
  evaluated_(false) {
  is_list_ = is_list_file(file_path);
}

TestBase::~TestBase(void) {
}

bool TestBase::evaluate(void) {
  std::cout << "### " << name_ << std::endl;
  success_ = this->run();
  std::cout << " Success: " << (success_?TEXT_OK:TEXT_FAIL) << std::endl;
  evaluated_ = true;
  return success_;
}

void TestBase::load_image(int load_flags, bool display_image, bool display_info) {
  this->load_image_to(file_path_,load_flags,image_);
  if(display_image) {
    this->display_image(image_,"Original image");
  }
  image_width_ = image_.cols;
  image_height_ = image_.rows;
  image_channels_ = image_.channels();
  image_size_ = image_width_ * image_height_ * image_channels_;
  if(display_info) {
    std::cout << "Image width: " << image_width_ << " px" << std::endl;
    std::cout << "Image height: " << image_height_ << " px" << std::endl;
    std::cout << "Image channels: " << image_channels_ << std::endl;
    std::cout << "Image size: " << image_size_ << " bytes" << std::endl;
  }
}

bool TestBase::load_image_dimensions(const std::size_t & width_default,
                                     const std::size_t & height_default) {
  if(!verify_path(file_path_)) {
    std::cout << " The specified file does not exist (" << file_path_ << ")" << std::endl;
    return false;
  }
  if(is_list_) {
    // this is a list file, read out the resolution from the first 2 lines
    std::ifstream list_file(file_path_);
    std::string list_line;
    for(int i=0;i<2;++i) {
      if(!std::getline(list_file,list_line)) {
        // error reading the line
        image_width_ = 0;
        image_height_ = 0;
      }
      if(i==0) {
        image_width_ = std::stoi(list_line);
      } else if(i==1) {
        image_height_ = std::stoi(list_line);
      }
    }
  } else if(file_path_ != NULL) {
    // this is a regular image
    cv::Mat temp_img;
    load_image_to(file_path_,cv::IMREAD_GRAYSCALE,temp_img);
    image_width_ = temp_img.cols;
    image_height_ = temp_img.rows;
  } else {
    assert(width_default > 0);
    assert(height_default > 0);
    // use the default arguments
    image_width_ = width_default;
    image_height_ = height_default;
  }
  return true;
}

void TestBase::load_image_to(const char * image_path,
                             int load_flags,
                             cv::Mat & dst_image) {
  // modify image path if it is not an absolute path
  std::string image_path_abs;
  if(image_path[0] != '/') {
    // relative path was given
    image_path_abs = get_executable_folder_path();
    image_path_abs += '/';
    image_path_abs += image_path;
  } else {
    // absolute path was given
    image_path_abs = image_path;
  }
  dst_image = cv::imread(image_path_abs.c_str(),load_flags);
}

void TestBase::save_image(const cv::Mat & image, const char * image_path) const {
  cv::imwrite(image_path,image);
}

void TestBase::display_image(const cv::Mat & image, const char * image_title) const {
  cv::imshow(image_title, image);
  cv::waitKey();
}

bool TestBase::compare_images(const cv::Mat & image1,
                              const cv::Mat & image2,
                              unsigned int diff_threshold,
                              bool display_difference,
                              bool display_difference_image,
                              int skip_first_n_rows,
                              int skip_first_n_cols,
                              int skip_last_n_rows,
                              int skip_last_n_cols) const {
  std::size_t difference_count = 0;
  cv::Mat difference_image;
  if(display_difference_image) {
    difference_image = cv::Mat(image1.rows,image1.cols,CV_8UC1);
  }
  bool success = true;
  do {
    if (image1.empty() && image2.empty()) {
      break;
    }
    if (image1.cols != image2.cols || image1.rows != image2.rows) {
      success = false;
      break;
    }
    for(int r=skip_first_n_rows;r<(image1.rows-skip_last_n_rows);++r) {
      for(int c=skip_first_n_cols;c<(image1.cols-skip_last_n_cols);++c) {
        unsigned int diff = std::abs(((int)image1.at<unsigned char>(r,c)) - ((int)image2.at<unsigned char>(r,c)));
        if(diff > diff_threshold) {
          success = false;
          if(display_difference) {
            std::cout << diff << std::endl;
          }
          if(display_difference_image) {
            difference_image.at<unsigned char>(r,c) = 0xFF;
          }
          ++difference_count;
        }
      }
    }
  } while(0);
  if(!success) {
    if(display_difference_image) {
      display_image(difference_image,"Difference image");
    }
    std::cout << " Difference count: " << difference_count << std::endl;
  }
  return success;
}

bool TestBase::compare_image_pyramid(const std::vector<cv::Mat> & image_pyramid1,
                                     const std::vector<cv::Mat> & image_pyramid2,
                                     unsigned int diff_threshold) const {
  if(image_pyramid1.size() != image_pyramid2.size()) {
    std::cout << " Number of pyramid levels differs (image_pyramid1=" <<
          image_pyramid1.size() << ",image_pyramid2=" << image_pyramid2.size() <<
          ")" << std::endl;
    return false;
  }
  for(unsigned int l=0;l<image_pyramid1.size();++l) {
    if(!this->compare_images(image_pyramid1[l],
                             image_pyramid2[l],
                             diff_threshold)) {
      std::cout << " Difference on level " << l << std::endl;
      return false;
    }
  }
  return true;
}

bool TestBase::run_benchmark(std::vector<Statistics> &,
                             std::vector<Statistics> &) {
  // Descendant test suites should add proper implementation
  return true;
}

bool TestBase::is_list_file(const char * file_path) {
  const char * list_extensions[] = {".txt"};
  const std::size_t list_extension_n = sizeof(list_extensions)/sizeof(char*);
  // find the extension
  const char * file_extension  = NULL;
  {
      const char * cur_file_path = file_path;
      while(cur_file_path != NULL) {
        file_extension = cur_file_path;
        cur_file_path = strchr(cur_file_path+1,'.');
      }
  }
  if(file_extension == NULL) {
    return false;
  }
  for(std::size_t i=0;i<list_extension_n;++i) {
    if(!strcmp(list_extensions[i],file_extension)) {
      return true;
    }
  }
  return false;
}

bool TestBase::verify_path(const char * path) {
  struct stat info;
  return (stat(path,&info) == 0);
}

bool TestBase::run_benchmark_suite(std::vector<Statistics> & stat_cpu,
                                   std::vector<Statistics> & stat_gpu) {
  bool result = true;
#if BENCHMARK_ENABLE
  DetectorBenchmark::init();
#endif /* BENCHMARK_ENABLE */
  if(is_list_) {
    // Evaluate the entire list instead of 1 image
    // file_path_ holds the path to the list

    std::ifstream list_file(file_path_);
    std::string list_line;
    unsigned int line_count = 0, image_count = 0;
    while(std::getline(list_file,list_line) && image_count < (unsigned int)max_image_num_) {
      // skip first 2 lines because they encode the image resolution
      ++line_count;
      if(line_count <= 2) {
        continue;
      }

      // load image
       ++image_count;
      load_image_to(list_line.c_str(),cv::IMREAD_GRAYSCALE,image_);

      // run benchmark
      result = run_benchmark(stat_cpu,stat_gpu) && result;
    }
  } else {
    // Load image 1x
    this->load_image(cv::IMREAD_GRAYSCALE,false,false);
    result = result && run_benchmark(stat_cpu,stat_gpu);
  }
  // Display all statistics
  if(stat_cpu.size()) {
    std::cout << " CPU ---" << std::endl;
    for(Statistics & stat : stat_cpu) {
      stat.display();
    }
  }
  if(stat_gpu.size()) {
    std::cout << " GPU ---" << std::endl;
    for(Statistics & stat : stat_gpu) {
      stat.display();
    }
  }
  // Display all benchmarks
#if BENCHMARK_ENABLE
  DetectorBenchmark::displayAll();
#endif /* BENCHMARK_ENABLE */
  return result;
}
