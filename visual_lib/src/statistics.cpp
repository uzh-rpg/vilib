/*
 * Statistics class storing and organizing statistical data
 * statistics.cpp
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
