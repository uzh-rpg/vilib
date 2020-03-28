/*
 * Benchmarking for feature detection tasks
 * detector_benchmark.cpp
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