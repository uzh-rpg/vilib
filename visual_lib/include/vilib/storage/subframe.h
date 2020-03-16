/*
 * Class for holding a particular representation of an input image
 * subframe.h
 */

#pragma once

#include <list>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime_api.h>
#ifdef ROS_SUPPORT
#include <sensor_msgs/Image.h>
#endif /* ROS_SUPPORT */

namespace vilib {

class Subframe {
public:
  enum class MemoryType : unsigned char {
    PAGED_HOST_MEMORY=0,
    PINNED_HOST_MEMORY,
    UNIFIED_MEMORY,
    LINEAR_DEVICE_MEMORY,
    PITCHED_DEVICE_MEMORY
  };

  Subframe(std::size_t width,
           std::size_t height,
           std::size_t data_bytes,
           MemoryType type);
  ~Subframe(void);
  void copy_from(const cv::Mat & h_img,
                 bool async = false,
                 cudaStream_t stream_num = 0);
#ifdef ROS_SUPPORT
  void copy_from(const sensor_msgs::ImageConstPtr & h_img,
                 bool async = false,
                 cudaStream_t stream_num = 0);
#endif /* ROS_SUPPORT */
  void copy_to(cv::Mat & h_img,
               bool async = false,
               cudaStream_t stream_num = 0) const;
  void display(void) const;

  std::size_t width_;       // width of a subframe in pixel units
  std::size_t height_;      // height of a subframe in pixel units
  std::size_t data_bytes_;  // representation length of one pixel in byte units
  MemoryType type_;    // underlying memory type
  std::size_t total_bytes_; // total used bytes
  std::size_t pitch_;       // length of a row in byte units
  unsigned char * data_;    // pointer of the buffer
};

} // namespace vilib
