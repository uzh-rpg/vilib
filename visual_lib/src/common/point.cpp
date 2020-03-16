/*
 * Point base class, that holds a triangulated point in the map
 * point_base.cpp
 */

#include "vilib/common/point.h"

namespace vilib {

std::mutex Point::last_id_mutex_;
std::size_t Point::last_id_ = 0;

std::size_t Point::getNewId(void) {
  std::lock_guard<std::mutex> lock(last_id_mutex_);
  return last_id_++;
}

} // namespace vilib
