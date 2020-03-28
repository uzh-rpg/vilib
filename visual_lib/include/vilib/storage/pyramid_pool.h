/*
 * Class for handling the allocation of entire image pyramids efficiently
 * pyramid_pool.h
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

#include <assert.h>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "vilib/storage/subframe_pool.h"
#include "vilib/common/types.h"

namespace vilib {

class PyramidPool {
public:
  PyramidPool(void) = delete;

  /*
   * Acquire an entire image pyramid from a preallocated buffer. If the buffer
   * got empty, a new allocation will take place.
   * @param preallocated_item_num number of frames preallocated per pool
   * @param original_width pixel width of the level0 (highest resolution) image
   * @param original_height pixel height of the level0 (highest resolution) image
   * @param data_bytes number of databytes each pixel requires (e.g.: grayscale = 1)
   * @param levels number of levels in the pyramid
   * @param memory_type the type of memory that is used for each frame
   * @param pyramid destination vector holding the image pyramid
   */
  static void get(std::size_t preallocated_item_num,
                  std::size_t original_width,
                  std::size_t original_height,
                  std::size_t data_bytes,
                  std::size_t levels,
                  Subframe::MemoryType memory_type,
                  std::vector<std::shared_ptr<Subframe>> & pyramid);

  /*
   * Return a previously acquired pyramid to the pyramid pool. The underlying
   * image frames will not be freed so that they can be reused.
   * @param pyramid the source vector holding the image pyramid
   */
  static void release(std::vector<std::shared_ptr<Subframe>> & pyramid);

  /*
   * Return a pyramid descriptor with image widhts, heights and pitches
   * @return pyramid descriptor
   */
  static inline pyramid_descriptor_t get_descriptor(void) {
    return desc_;
  }

  /*
   * Preallocate an entire image pyramid based on the level0 image width and
   * image height. One should not call this function directly.
   * @param preallocated_item_num number of frames preallocated per pool
   * @param original_width level0 image width
   * @param original_height level0 image height
   * @param data_bytes number of bytes each pixel requires
   * @param levels number of levels that get preallocated with decreasing resolutions
   *               each level halves the resolution of the preceding level
   * @param memory_type the type of memory that is used for each frame
   */
  static void init(std::size_t preallocated_item_num,
                   std::size_t original_width,
                   std::size_t original_height,
                   std::size_t data_bytes,
                   std::size_t levels,
                   Subframe::MemoryType memory_type);

  /*
   * Deinitialize the entire pyramid pool. This function is only for testing
   * and benchmarking purposes.
   */
  static void deinit(void);

  /*
   * Return the image pyramid level count
   */
  static inline std::size_t getCurrentLevelNum(void) {
    return pool_.size();
  }

  /*
   * Return the currently allocated subframe count
   */
  static inline std::size_t getCurrentSubframeCount(void) {
    assert(pool_.size() > 0);
    return pool_[0]->preallocated_num();
  }

private:
  static bool initialized_;
  static std::vector<std::shared_ptr<SubframePool>> pool_;
  static pyramid_descriptor_t desc_;
};

} // namespace vilib
