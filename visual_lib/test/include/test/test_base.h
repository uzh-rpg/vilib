/*
 * Base class for testing various functionalities
 * test_base.h
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

#pragma once

#include <memory>
#include <functional>
#include <opencv2/core/mat.hpp>
#include "vilib/timer.h"
#include "vilib/statistics.h"

class TestBase;

// Macro for define -> C-string conversion
#define STRINGIFY(s)                         STRINGIFY_2(s)
#define STRINGIFY_2(s)                       #s

struct TestCase {
    std::unique_ptr<TestBase> test_;
    std::size_t rep_cnt_;
    std::function<bool(const std::unique_ptr<TestBase> &)> fn_;

    TestCase(TestBase * test_ptr):
        test_(test_ptr), rep_cnt_(1), fn_(nullptr) {}
    TestCase(TestBase * test_ptr, std::size_t rep_cnt):
        test_(test_ptr), rep_cnt_(rep_cnt), fn_(nullptr) {}
    TestCase(TestBase * test_ptr, std::size_t rep_cnt, std::function<bool(const std::unique_ptr<TestBase> &)> fn):
        test_(test_ptr), rep_cnt_(rep_cnt), fn_(fn) {}
};

class TestBase {
public:
    TestBase(const char * name, const char * file_path, const int max_image_num = -1);
    ~TestBase(void);
    bool evaluate(void);
protected:
    virtual bool run(void) = 0;
    bool run_benchmark_suite(std::vector<vilib::Statistics> & stat_cpu,
                             std::vector<vilib::Statistics> & stat_gpu);
    virtual bool run_benchmark(std::vector<vilib::Statistics> & stat_cpu,
                               std::vector<vilib::Statistics> & stat_gpu);
    void load_image(int load_flags, bool display_image, bool display_info);
    void load_image_to(const char * image_path, int load_flags, cv::Mat & image);
    void display_image(const cv::Mat & image, const char * image_title) const;
    void save_image(const cv::Mat & image, const char * image_path) const;
    bool compare_images(const cv::Mat & image1,
                        const cv::Mat & image2,
                        unsigned int diff_threshold,
                        bool display_difference=false,
                        bool display_difference_image=false,
                        int skip_first_n_rows=0,
                        int skip_first_n_cols=0,
                        int skip_last_n_rows=0,
                        int skip_last_n_cols=0) const;
    bool compare_image_pyramid(const std::vector<cv::Mat> & image_pyramid1,
                               const std::vector<cv::Mat> & image_pyramid2,
                               unsigned int diff_threshold) const;
    bool is_list_file(const char * file_path);
    bool verify_path(const char * path);
    bool load_image_dimensions(const std::size_t & width_default = 0,
                               const std::size_t & height_default = 0);

    const char * name_;
    const char * file_path_;
    const int max_image_num_;
    bool is_list_;
    cv::Mat image_;
    unsigned int image_width_;
    unsigned int image_height_;
    unsigned int image_channels_;
    std::size_t image_size_;
    bool success_;
    bool evaluated_;
};
