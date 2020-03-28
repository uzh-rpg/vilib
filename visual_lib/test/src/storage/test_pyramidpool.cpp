/*
 * Tests for pyramid pool functionalities
 * test_pyramidpool.cpp
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

#include <memory>
#include "vilib/storage/pyramid_pool.h"
#include "vilib/storage/subframe.h"
#include "vilib/timer.h"
#include "test/storage/test_pyramidpool.h"
#include "test/common.h"

using namespace vilib;

#define IMAGE_SIZE_WIDTH                    752
#define IMAGE_SIZE_HEIGHT                   480
#define IMAGE_MEMORY_TYPE                   (Subframe::MemoryType::PITCHED_DEVICE_MEMORY)
#define IMAGE_PYRAMID_LEVELS                5
#define PYRAMIDPOOL_PREALLOCATED_NUM        10
#define PYRAMIDPOOL_REQUESTED_NUM           30
#define ITERATION_NUM                       100

TestPyramidPool::TestPyramidPool(void) :
 TestBase("PyramidPool", NULL) {
}

TestPyramidPool::~TestPyramidPool(void) {
}

bool TestPyramidPool::run(void) {
  bool success = true;
  std::size_t width = IMAGE_SIZE_WIDTH;
  std::size_t height = IMAGE_SIZE_HEIGHT;
  std::size_t n_pyramid_levels = IMAGE_PYRAMID_LEVELS;
  std::size_t data_bytes = sizeof(unsigned char);
  Subframe::MemoryType type = IMAGE_MEMORY_TYPE;

  Timer pool_time("Pool creation (with " STRINGIFY(PYRAMIDPOOL_PREALLOCATED_NUM) " frames)");
  Timer prealloc_time("Preallocated access");
  Timer newalloc_time("New allocation");

  for(std::size_t i=0;i<ITERATION_NUM;++i) {
    pool_time.start();
    PyramidPool::init(PYRAMIDPOOL_PREALLOCATED_NUM,
                      width,
                      height,
                      data_bytes,
                      n_pyramid_levels,
                      type);
    pool_time.stop();
    pool_time.add_to_stat_n_reset();
    success = success && (PyramidPool::getCurrentLevelNum() == n_pyramid_levels);

    // Vector for holding
    std::vector<std::vector<std::shared_ptr<Subframe>>> pyramids;

    // access PYRAMIDPOOL_REQUESTED_NUM elements
    for(std::size_t i=0;i<PYRAMIDPOOL_REQUESTED_NUM;++i) {
      pyramids.push_back(std::vector<std::shared_ptr<Subframe>>());
      if(i < PYRAMIDPOOL_PREALLOCATED_NUM) {
        prealloc_time.start();
        PyramidPool::get(PYRAMIDPOOL_PREALLOCATED_NUM,
                         width,
                         height,
                         data_bytes,
                         n_pyramid_levels,
                         type,
                         pyramids.back());
        prealloc_time.stop();
        prealloc_time.add_to_stat_n_reset();
      } else {
        newalloc_time.start();
        PyramidPool::get(PYRAMIDPOOL_PREALLOCATED_NUM,
                         width,
                         height,
                         data_bytes,
                         n_pyramid_levels,
                         type,
                         pyramids.back());
        newalloc_time.stop();
        newalloc_time.add_to_stat_n_reset();
      }
    }

    // return them to the originating pool
    for(std::size_t i=0;i<PYRAMIDPOOL_REQUESTED_NUM;++i) {
      success = success && (PyramidPool::getCurrentSubframeCount() == i);
      PyramidPool::release(pyramids[i]);
    }
    pyramids.clear();

    // deinit pyramid pool
    PyramidPool::deinit();
  }

  //Statistics
  pool_time.display_stat_usec();
  prealloc_time.display_stat_usec();
  newalloc_time.display_stat_usec();
  return success;
}
