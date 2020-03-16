/*
 * High-level functions for image preprocessing
 * image_preprocessing.h
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
