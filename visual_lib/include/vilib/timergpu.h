/*
 * TimerGPU class for profiling algorithms on the GPU
 * timergpu.h
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
