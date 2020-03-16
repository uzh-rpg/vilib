/*
 * Tests for image pyramid functionalities
 * test_pyramid.h
 */

#pragma once

#include "test/test_base.h"

class TestPyramid : public TestBase {
public:
  TestPyramid(const char * image_path);
  ~TestPyramid(void);
protected:
  bool run(void);
  void preallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image,
                               std::vector<std::size_t> & pyramid_width,
                               std::vector<std::size_t> & pyramid_height,
                               std::vector<std::size_t> & pyramid_pitch,
                               std::size_t pyramid_levels);
  void preallocate_pyramid_cpu(std::vector<cv::Mat> & pyramid_image_cpu,
                               std::size_t pyramid_levels);
  void deallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image);
  void copy_pyramid_from_gpu(std::vector<unsigned char*> & pyramid_image_gpu,
                             std::vector<cv::Mat> & pyramid_image_cpu,
                             std::vector<std::size_t> & pyramid_pitch_gpu,
                             std::size_t pyramid_levels);
};
