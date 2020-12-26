/*
 * Gradient image (utility class for debugging GPU code)
 * gradient_image.cpp
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
