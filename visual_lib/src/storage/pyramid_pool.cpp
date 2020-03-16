/*
 * Class for handling the allocation of entire image pyramids efficiently
 * pyramid_pool.cpp
 */

#include <assert.h>
#include "vilib/storage/pyramid_pool.h"
#include "vilib/storage/subframe.h"
#include "vilib/storage/subframe_pool.h"

namespace vilib {

bool PyramidPool::initialized_ = false;
std::vector<std::shared_ptr<SubframePool>> PyramidPool::pool_;
pyramid_descriptor_t PyramidPool::desc_;

void PyramidPool::init(std::size_t preallocated_item_num,
                       std::size_t original_width,
                       std::size_t original_height,
                       std::size_t data_bytes,
                       std::size_t levels,
                       Subframe::MemoryType memory_type) {
  /*
   * Note to future self:
   * make sure that the original image width & height are divisible by 2
   * for all levels.
   */
  assert(levels > 0);
  assert((original_width %(1<<(levels-1))) == 0);
  assert((original_height%(1<<(levels-1))) == 0);
  for(unsigned int l=0;l<levels;++l) {
    std::size_t level_width  = original_width >> l;
    std::size_t level_height = original_height >> l;
    pool_.push_back(std::shared_ptr<SubframePool>(
                     new SubframePool(preallocated_item_num,
                                      level_width,
                                      level_height,
                                      data_bytes,
                                      memory_type)));
    // get description info
    desc_.w[l] = pool_[l]->get_width();
    desc_.h[l] = pool_[l]->get_height();
    desc_.p[l] = pool_[l]->get_pitch();
  }
  desc_.l = levels;
  initialized_ = true;
}

void PyramidPool::deinit(void) {
  pool_.clear();
  initialized_ = false;
}

void PyramidPool::get(std::size_t preallocated_item_num,
                      std::size_t original_width,
                      std::size_t original_height,
                      std::size_t data_bytes,
                      std::size_t levels,
                      Subframe::MemoryType memory_type,
                      std::vector<std::shared_ptr<Subframe>> & pyramid) {
  /*
   * Note to future self:
   * 0) Initialize the subframe pools if necessary
   *    Hence, the first frame processing might take longer than the rest
   * 1) Give back subframes for all pyramid levels
   */
  if(!initialized_) {
    init(preallocated_item_num,
         original_width,
         original_height,
         data_bytes,
         levels,
         memory_type);
  }
  for(std::size_t l=0;l<levels;++l) {
    pyramid.push_back(pool_[l]->get_subframe());
  }
}

void PyramidPool::release(std::vector<std::shared_ptr<Subframe>> & pyramid) {
  if(!initialized_) return;
  for(std::size_t l=0;l<pool_.size();++l) {
    pool_[l]->return_subframe(pyramid[l]);
  }
}

} // namespace vilib
