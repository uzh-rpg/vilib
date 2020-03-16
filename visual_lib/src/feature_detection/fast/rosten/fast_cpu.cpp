/*
 * FAST feature detector on the CPU (as provided by Edward Rosten)
 * fast_cpu.cpp
 */

#include <assert.h>
#include <algorithm>
#include "vilib/feature_detection/fast/rosten/fast_cpu.h"
#include "fast.h"
#include <iostream>

namespace vilib {
namespace rosten {

// Minimum border: 3, this is the minimum border needed because of the Bresenham circle
#define MINIMUM_BORDER                3

template <bool use_grid>
FASTCPU<use_grid>::FASTCPU(
                 const std::size_t image_width,
                 const std::size_t image_height,
                 const std::size_t cell_size_width,
                 const std::size_t cell_size_height,
                 const std::size_t min_level,
                 const std::size_t max_level,
                 const std::size_t horizontal_border,
                 const std::size_t vertical_border,
                 const float threshold,
                 const int min_arc_length,
                 const fast_score score) :
  DetectorBase(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               std::max((std::size_t)MINIMUM_BORDER,horizontal_border),
               std::max((std::size_t)MINIMUM_BORDER,vertical_border)),
   threshold_(threshold) {
  assert(min_arc_length >= 9 && min_arc_length <= 12);
  switch(score) {
    case SUM_OF_ABS_DIFF_ON_ARC:
      if(min_arc_length == 9) {
        fn_ = fast9_detect_nonmax<false>;
      } else if(min_arc_length == 10) {
        fn_ = fast10_detect_nonmax<false>;
      } else if(min_arc_length == 11) {
        fn_ = fast11_detect_nonmax<false>;
      } else if(min_arc_length == 12) {
        fn_ = fast12_detect_nonmax<false>;
      }
      break;
    case MAX_THRESHOLD:
      if(min_arc_length == 9) {
        fn_ = fast9_detect_nonmax<true>;
      } else if(min_arc_length == 10) {
        fn_ = fast10_detect_nonmax<true>;
      } else if(min_arc_length == 11) {
        fn_ = fast11_detect_nonmax<true>;
      } else if(min_arc_length == 12) {
        fn_ = fast12_detect_nonmax<true>;
      }
      break;
    default:
      assert(0);
  }
}

template <bool use_grid>
FASTCPU<use_grid>::~FASTCPU(void) {
}

template <bool use_grid>
std::size_t FASTCPU<use_grid>::count(void) const {
  if(use_grid) {
    return grid_.getOccupiedCount();
  } else {
    return keypoints_.size();
  }
}

template <bool use_grid>
void FASTCPU<use_grid>::reset(void) {
  if(use_grid) {
    grid_.reset();
  } else {
    keypoints_.clear();
  }
}

template <bool use_grid>
void FASTCPU<use_grid>::detect(const std::vector<cv::Mat> & image_pyramid) {
  for(unsigned int l=min_level_;l<image_pyramid.size() && l<max_level_;++l) {
    int corner_num = 0;
    xys * corners = fn_(image_pyramid[l].data,
                       image_pyramid[l].cols,
                       image_pyramid[l].rows,
                       static_cast<int>(image_pyramid[l].step),
                       threshold_,
                       &corner_num);

    const int maxw = image_pyramid[l].cols-horizontal_border_;
    const int maxh = image_pyramid[l].rows-vertical_border_;
    for(int i=0;i<corner_num;++i) {
      if(corners[i].x < (int)horizontal_border_ ||
         corners[i].y < (int)vertical_border_ ||
         corners[i].x >= maxw ||
         corners[i].y >= maxh) {
        continue;
      }

      double scale = (double)(1<<l);
      double sub_x = corners[i].x*scale;
      double sub_y = corners[i].y*scale;
      double score = corners[i].s;

      if(use_grid) {
        addFeaturePoint(sub_x,sub_y,score,l);
      } else {
        keypoints_.emplace_back(sub_x,sub_y,score,l);
      }
    }
    free(corners);
  }
}

// Explicit instantiations
template class FASTCPU<false>;
template class FASTCPU<true>;

} // namespace rosten
} // namespace vilib
