/*
 * Point base class, that holds a triangulated point in the map
 * point_base.cpp
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

#include "vilib/common/point.h"

namespace vilib {

std::mutex Point::last_id_mutex_;
std::size_t Point::last_id_ = 0;

std::size_t Point::getNewId(void) {
  std::lock_guard<std::mutex> lock(last_id_mutex_);
  return last_id_++;
}

} // namespace vilib
