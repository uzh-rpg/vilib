/*
 * High-level functions for image preprocessing
 * image_preprocessing.h
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

#include <vector>
#include <memory>
#include <opencv2/core/mat.hpp>
#ifdef ROS_SUPPORT
#include <sensor_msgs/Image.h>
#endif /* ROS_SUPPORT */
#include "vilib/storage/subframe.h"

namespace vilib {

void preprocess_image(const cv::Mat & img,
                      std::vector<std::shared_ptr<Subframe>> & pyramid,
                      cudaStream_t stream);

#ifdef ROS_SUPPORT
void preprocess_image(const sensor_msgs::ImageConstPtr & msg,
                      std::vector<std::shared_ptr<Subframe>> & pyramid,
                      cudaStream_t stream);
#endif /* ROS_SUPPORT */

} // namespace vilib
