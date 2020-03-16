/*
 * FAST feature detector on the CPU (as provided by Edward Rosten)
 * fast_cpu.h
 */

#pragma once

#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/fast/fast_common.h"

namespace vilib {
namespace rosten {

struct xys_tuple;

template <bool use_grid>
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
          const float threshold,
          const int min_arc_length,
          const fast_score score);
  ~FASTCPU(void);
  void detect(const std::vector<cv::Mat> & image) override;
  std::size_t count(void) const override;
  void reset(void);
private:
  float threshold_;
  struct xys_tuple * (*fn_)(const unsigned char * im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
};

} // namespace rosten
} // namespace vilib
