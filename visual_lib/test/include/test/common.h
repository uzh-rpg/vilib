/*
 * Common functionalities within the test suite
 * common.h
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
