/*
 * Timer class for profiling algorithms on the CPU
 * timer.h
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
