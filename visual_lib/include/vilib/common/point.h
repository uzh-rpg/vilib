/*
 * Point class for holding triangulated 3D points
 * point.h
 */

#pragma once

#include <mutex>

namespace vilib {

class Point {
public:
  static std::size_t getNewId(void);
private:
  static std::mutex last_id_mutex_;
  static std::size_t last_id_;
};

} // namespace vilib