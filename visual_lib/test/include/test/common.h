/*
 * Common functionalities within the test suite
 * common.h
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

#include <iostream>
#include <functional>
#include <opencv2/core/mat.hpp>

#define TEXT_OK     "OK"
#define TEXT_FAIL   "FAIL"

template<class T>
bool mat_compare(const cv::Mat & mat1,
                 const cv::Mat & mat2,
                 std::function<bool(const T &, const T &)> predicate) {
  if (mat1.empty() && mat2.empty()) {
    return true;
  }
  if (mat1.cols != mat2.cols || mat1.rows != mat2.rows) {
    return false;
  }
  const T * data1 = mat1.data;
  const T * data2 = mat2.data;
  std::size_t mat_data_cnt = mat1.rows * mat1.cols;
  bool success = true;
  for(std::size_t idx=0;idx<mat_data_cnt;++idx) {
    if(!predicate(data1[idx],data2[idx])) {
#if 0
      unsigned int r = idx/mat1.cols;
      unsignet int c = (idx - r*mat1.cols);
      std::cout << "(r,c) = (" << r << "," << c << ")" << std::endl;
#endif /* 0 */
      success = false;
    }
  }
  return success;
}

template<class T>
bool array_compare(const T * a1,
                   const T * a2,
                   unsigned int a_size,
                   std::function<bool(const T &, const T &)> predicate) {
  bool success = true;
  for(unsigned int i=0;i<a_size;++i) {
    if(!predicate(a1[i],a2[i])) {
      success = false;
    }
  }
  return success;
}
