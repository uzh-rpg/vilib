/*
 * Gradient image (utility class for debugging GPU code)
 * gradient_image.h
 */

#pragma once

#include <opencv2/core/mat.hpp>

class GradientImage {
public:
  enum class Type {
    HORIZONTAL,
    VERTICAL
  };

  GradientImage(const std::size_t & width, const std::size_t & height, Type type);
  ~GradientImage(void) = default;

  void display(void) const;

  inline cv::Mat & get(void) {
    return image_;
  }

  inline const cv::Mat & get(void) const {
    return image_;
  }

  inline std::size_t width(void) const {
    return image_.cols;
  }

  inline std::size_t height(void) const {
    return image_.rows;
  }
private:
  cv::Mat image_;
};
