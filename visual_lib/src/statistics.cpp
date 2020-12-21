/*
 * Statistics class storing and organizing statistical data
 * statistics.cpp
 * 
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
