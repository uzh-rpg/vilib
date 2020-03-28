/*
 * Gradient image (utility class for debugging GPU code)
 * gradient_image.cpp
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

#include <opencv2/highgui.hpp>
#include "test/groundtruth/gradient_image.h"

GradientImage::GradientImage(const std::size_t & width,
                             const std::size_t & height,
                             Type type) {
  image_ = cv::Mat(height,width,CV_8UC1);
  // initialize with a gradient pattern
  unsigned char * data_ptr = image_.data;
  unsigned char data = 0x00;
  for(std::size_t r=0;r<height;++r,data_ptr+=image_.step - width) {
    if(type == Type::HORIZONTAL) {
      data = 0x00;
    }
    for(std::size_t c=0;c<width;++c,++data_ptr) {
      *data_ptr = data;
      if(type == Type::HORIZONTAL) {
        ++data;
      }
    }
    if(type == Type::VERTICAL) {
      ++data;
    }
  }
}

void GradientImage::display(void) const {
  cv::imshow("Gradient test image", image_);
  cv::waitKey();
}
