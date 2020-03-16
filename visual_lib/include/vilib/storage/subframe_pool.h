/*
 * Functions for handling preallocated memory-pools
 * subframe_pool.h
 */

#pragma once

#include <mutex>
#include <memory>
#include <forward_list>
#include "vilib/storage/subframe.h"

namespace vilib {

class SubframePool {
public:
  SubframePool(std::size_t preallocated_item_num,
               std::size_t width,
               std::size_t height,
               std::size_t data_bytes,
               Subframe::MemoryType type);
  ~SubframePool(void);
  std::shared_ptr<Subframe> get_subframe(void);
  void return_subframe(std::shared_ptr<Subframe> frame);
  std::size_t preallocated_num(void);
  inline std::size_t get_width(void) const  { return width_; }
  inline std::size_t get_height(void) const { return height_; }
  inline std::size_t get_pitch(void) const  { return pitch_; }
private:
  std::forward_list<std::shared_ptr<Subframe>> items_;
  std::size_t items_size_;
  std::mutex items_mutex_;
  std::size_t width_;
  std::size_t height_;
  std::size_t pitch_;
  std::size_t data_bytes_;
  Subframe::MemoryType type_;
};

} // namespace vilib
