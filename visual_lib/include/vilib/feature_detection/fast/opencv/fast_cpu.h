/*
 * FAST feature detector on the CPU (as provided by OpenCV)
 * fast_cpu.h
 */

#pragma once

#include <opencv2/features2d.hpp>
#include "vilib/feature_detection/detector_base.h"

namespace vilib {
namespace opencv { 

class FASTCPU : public DetectorBase {
public:
  FASTCPU(const std::size_t image_width,
          const std::size_t image_height,
          const std::size_t cell_size_width,
          const std::size_t cell_size_height,
          const std::size_t min_level,
          const std::size_t max_level,
          const std::size_t horizontal_border,
          const std::size_t vertical_border,
          const float threshold);
  ~FASTCPU(void);
  void detect(const std::vector<cv::Mat> & image) override;
private:
  float threshold_;
  cv::Ptr<cv::Feature2D> detector_;
};

} // namespace opencv
} // namespace vilib
