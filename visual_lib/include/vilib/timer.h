/*
 * Timer class for profiling algorithms on the CPU
 * timer.h
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

#include "vilib/statistics.h"

namespace vilib {

class Timer {
public:
  Timer(const char * name = nullptr, int indent = 1);
  Timer(const std::string & name, int indent = 1);

  void start(void);
  void stop(void);
  void pause(void);
  void cont(void);
  void reset(void);
  void add_to_stat_n_reset(void);
  double elapsed_usec(void) const { return time_; }
  double elapsed_msec(void) const { return time_/1.0e3; }
  double elapsed_sec(void) const { return time_/1.0e6; }

  void display_usec(void) const;
  void display_stat_usec(void) const;
private:
  double get_localtime_usec(void) const;

  std::string name_;
  double time_; // time is expressed in usec
  int indent_;
  Statistics stat_;

  static const int name_size_characters_ = 35;
  static const int time_size_characters_ = 15;
};

} // namespace vilib
