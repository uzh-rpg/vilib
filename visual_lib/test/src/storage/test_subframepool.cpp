/*
 * Tests for subframe pool functionalities
 * test_subframepool.cpp
 */

#include <memory>
#include "vilib/storage/subframe_pool.h"
#include "vilib/storage/subframe.h"
#include "vilib/timer.h"
#include "test/storage/test_subframepool.h"
#include "test/common.h"

using namespace vilib;

#define IMAGE_SIZE_WIDTH                752
#define IMAGE_SIZE_HEIGHT               480
#define IMAGE_MEMORY_TYPE               (Subframe::MemoryType::PITCHED_DEVICE_MEMORY)
#define POOL_PREALLOCATED_NUM           10
#define POOL_REQUESTED_NUM              30
#define REPETITION_COUNT                100

TestSubframePool::TestSubframePool(void) :
 TestBase("SubframePool", NULL) {
}

TestSubframePool::~TestSubframePool(void) {
}

bool TestSubframePool::run(void) {
  bool success = true;
  std::size_t width = IMAGE_SIZE_WIDTH;
  std::size_t height = IMAGE_SIZE_HEIGHT;
  std::size_t data_bytes = sizeof(unsigned char);
  Subframe::MemoryType type = IMAGE_MEMORY_TYPE;

  Timer pool_time("Pool creation (with " STRINGIFY(POOL_PREALLOCATED_NUM) " frames)");
  Timer prealloc_time("Preallocated access");
  Timer newalloc_time("New allocation");

  for(std::size_t i=0;i<REPETITION_COUNT;++i) {
    pool_time.start();
    std::unique_ptr<SubframePool> pool_images(
      new SubframePool(POOL_PREALLOCATED_NUM,width,height,data_bytes,type));
    pool_time.stop();
    pool_time.add_to_stat_n_reset();

    std::vector<std::shared_ptr<Subframe>> subframes;
    // access POOL_REQUESTED_NUM elements
    for(std::size_t i=0;i<POOL_REQUESTED_NUM;++i) {
      if(i < POOL_PREALLOCATED_NUM) {
        prealloc_time.start();
        std::shared_ptr<Subframe> subframe_prealloc = pool_images->get_subframe();
        prealloc_time.stop();
        prealloc_time.add_to_stat_n_reset();
        subframes.push_back(subframe_prealloc);
      } else {
        newalloc_time.start();
        std::shared_ptr<Subframe> subframe_new = pool_images->get_subframe();
        newalloc_time.stop();
        newalloc_time.add_to_stat_n_reset();
        subframes.push_back(subframe_new);
      }
    }
    // return them to the originating pool
    for(std::size_t i=0;i<POOL_REQUESTED_NUM;++i) {
      success = success && (pool_images->preallocated_num()==i);
      pool_images->return_subframe(subframes[i]);
    }
    subframes.clear();
  }

  //Statistics
  pool_time.display_stat_usec();
  prealloc_time.display_stat_usec();
  newalloc_time.display_stat_usec();
  return success;
}
