/*
 * TimerGPU class for profiling algorithms on the GPU
 * timergpu.h
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

#pragma once

#include <cuda.h>
#include "vilib/statistics.h"

namespace vilib {

class TimerGPU {
public:
  TimerGPU(const char * name, int indent = 1);
  TimerGPU(const std::string & name, int indent = 1);
  ~TimerGPU(void);

  void start(cudaStream_t stream = 0);
  void stop(bool synchronize = true, cudaStream_t stream = 0);
  void sync(void);
  void add_to_stat_n_reset(void);
  double elapsed_usec(void) const { return time_; }
  double elapsed_sec(void) const { return time_/1.0e6; }

  void display_usec(void) const;
  void display_stat_usec(void) const;
private:

  std::string name_;
  double time_; // time is expressed in usec
  int indent_;
  Statistics stat_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;

  static const int name_size_characters_ = 35;
  static const int time_size_characters_ = 15;
};

} // namespace vilib
