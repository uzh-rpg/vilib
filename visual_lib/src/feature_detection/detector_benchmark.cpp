/*
 * Benchmarking for feature detection tasks
 * detector_benchmark.cpp
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

#include "vilib/feature_detection/detector_benchmark.h"
#include "vilib/cuda_common.h"

namespace vilib {

std::unique_ptr<DetectorBenchmark> DetectorBenchmark::inst_ = nullptr;

DetectorBenchmark::DetectorBenchmark(void) {
    host_timers_.reserve(3);
    host_timers_.emplace_back("Reset",2);
    host_timers_.emplace_back("Upload",2);
    host_timers_.emplace_back("Download",2);
    /*
     * Note to future self:
     * it seems like that cudaEvents are not behaving well
     * if copied after a vector resize.
     * Hence we need to reserve the proper vector size apriori.
     */
    device_timers_.reserve(3);
    device_timers_.emplace_back("Pyramid",2);
    device_timers_.emplace_back("CRF",2);
    device_timers_.emplace_back("NMS",2);
}

void DetectorBenchmark::init(void) {
    if(inst_ != nullptr) return;
    inst_.reset(new DetectorBenchmark());
}

void DetectorBenchmark::startHost(Host type, bool sync_device_before) {
    if(inst_ == nullptr) return;
    if(sync_device_before) {
        CUDA_API_CALL(cudaDeviceSynchronize());
    }
    inst_->host_timers_[static_cast<std::size_t>(type)].start();
}

void DetectorBenchmark::stopHost(Host type) {
    if(inst_ == nullptr) return;
    Timer & tim = inst_->host_timers_[static_cast<std::size_t>(type)]; 
    tim.stop();
    tim.add_to_stat_n_reset();
}

void DetectorBenchmark::startDevice(Device type, cudaStream_t stream) {
    if(inst_ == nullptr) return;
    inst_->device_timers_[static_cast<std::size_t>(type)].start(stream);
}

void DetectorBenchmark::stopDevice(Device type, cudaStream_t stream) {
    if(inst_ == nullptr) return;
    inst_->device_timers_[static_cast<std::size_t>(type)].stop(false,stream);
}

void DetectorBenchmark::collectDevice(Device type) {
    if(inst_ == nullptr) return;
    TimerGPU & tim = inst_->device_timers_[static_cast<std::size_t>(type)];
    tim.sync();
    tim.add_to_stat_n_reset();
}

void DetectorBenchmark::displayAll(void) {
    if(inst_ == nullptr) return;
    // Host Timers
    for(std::size_t i=0;i<inst_->host_timers_.size();++i) {
      inst_->host_timers_[i].display_stat_usec();
    }
    // Device Timers
    for(std::size_t i=0;i<inst_->device_timers_.size();++i) {
      inst_->device_timers_[i].display_stat_usec();
    }
}

} // namespace vilib