/*
 * Timer class for profiling algorithms on the GPU
 * timergpu.cpp
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

#include <iostream>
#include <iomanip>
#include "vilib/cuda_common.h"
#include "vilib/timergpu.h"

namespace vilib {

TimerGPU::TimerGPU(const char * name, int indent) :
  name_(name),time_(0.0),indent_(indent),stat_("usec") {
  CUDA_API_CALL(cudaEventCreate(&start_event_));
  CUDA_API_CALL(cudaEventCreate(&stop_event_));
}

TimerGPU::TimerGPU(const std::string & name, int indent) :
  name_(name),time_(0.0),indent_(indent),stat_("usec") {
  CUDA_API_CALL(cudaEventCreate(&start_event_));
  CUDA_API_CALL(cudaEventCreate(&stop_event_));
}

TimerGPU::~TimerGPU(void) {
  CUDA_API_CALL(cudaEventDestroy(start_event_));
  CUDA_API_CALL(cudaEventDestroy(stop_event_));
}

void TimerGPU::start(cudaStream_t stream) {
  CUDA_API_CALL(cudaEventRecord(start_event_,stream));
}

void TimerGPU::stop(bool synchronize, cudaStream_t stream) {
  CUDA_API_CALL(cudaEventRecord(stop_event_,stream));
  if(synchronize) {
    sync();
  }
}

void TimerGPU::sync(void) {
  CUDA_API_CALL(cudaEventSynchronize(stop_event_));
  float time_ms;
  CUDA_API_CALL(cudaEventElapsedTime(&time_ms,start_event_,stop_event_));
  /*
   * Note to future self and others:
   * according to NVIDIA, the elapsed time is expressed in ms,
   * with 0.5usec resolution.
   */
  time_ = time_ms * 1.0e3; // go from ms to us
}

void TimerGPU::add_to_stat_n_reset(void) {
  stat_.add(time_);
  time_ = 0.0;
}

void TimerGPU::display_usec(void) const {
  for(int i=0;i<indent_;++i) {
    std::cout << " ";
  }
  std::cout << std::setw(name_size_characters_) << std::left << name_ << ": "
            << std::setw(time_size_characters_) << std::right << time_ << " usec" << std::endl;
}

void TimerGPU::display_stat_usec(void) const {
  for(int i=0;i<indent_;++i) {
    std::cout << " ";
  }
  std::cout << std::setw(name_size_characters_) << std::left  << name_ << ": ";
  stat_.display();
}

} // namespace vilib
