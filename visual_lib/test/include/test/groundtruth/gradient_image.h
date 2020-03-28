/*
 * Gradient image (utility class for debugging GPU code)
 * gradient_image.h
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
