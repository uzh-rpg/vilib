/*
 * Statistics class storing and organizing statistical data
 * statistics.cpp
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

#include <limits>
#include <iostream>
#include "vilib/statistics.h"

namespace vilib {

Statistics::Statistics(const char * unit, const char * name, int indent) {
  unit_ = unit;
  if(name != NULL) {
    name_ = name;
  }
  indent_ = indent;
  reset();
}

void Statistics::add(double data) {
  if(data < min_) {
    min_ = data;
  }
  if(data > max_) {
    max_ = data;
  }
  avg_ += data;
  avg_count_ += 1;
}

void Statistics::reset(void) {
  min_ = std::numeric_limits<double>::max();
  max_ = std::numeric_limits<double>::min();
  avg_ = 0;
  avg_count_ = 0;
}

void Statistics::display(int decimal_places, double scale, const char * new_unit) const {
  if(name_.length() > 0) {
    for(int i=0;i<indent_;++i) {
      std::cout << " ";
    }
    std::cout << name_ << ": ";
  }
  int decimal_places_old = std::cout.precision();
  if(decimal_places) {
    std::cout.precision(decimal_places);
  }
  if(avg_count_ == 0) {
    std::cout << "min: " << "-"
              << ", max: " << "-"
              << ", avg: " << "-"
              << " " << (new_unit?new_unit:unit_)
              << std::endl;
  } else {
    std::cout << "min: " << (min_*scale)
              << ", max: " << (max_*scale)
              << ", avg: " << (avg_*scale/avg_count_)
              << " " << (new_unit?new_unit:unit_)
              << std::endl;
  }
  if(decimal_places) {
    std::cout.precision(decimal_places_old);
  }
}

} // namespace vilib
