/*
 * Gradient image (utility class for debugging GPU code)
 * gradient_image.cpp
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
