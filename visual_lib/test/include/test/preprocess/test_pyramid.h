/*
 * Tests for image pyramid functionalities
 * test_pyramid.h
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
