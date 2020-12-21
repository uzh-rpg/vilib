/*
 * Benchmarking for feature detection tasks
 * detector_benchmark.h
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

#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include "vilib/timer.h"
#include "vilib/timergpu.h"
#include "vilib/benchmark.h"

namespace vilib {

class DetectorBenchmark {
public:
    enum class Host {
        Reset=0,
        Upload,
        Download
    };
    enum class Device {
        Pyramid=0,
        CRF,
        NMS
    };

    static void init(void);
    static void startHost(Host type, bool sync_device_before);
    static void stopHost(Host type);
    static void startDevice(Device type, cudaStream_t stream);
    static void stopDevice(Device type, cudaStream_t stream);
    static void collectDevice(Device type);
    static void displayAll(void);
private:
    DetectorBenchmark(void);

    static std::unique_ptr<DetectorBenchmark> inst_;
    std::vector<TimerGPU> device_timers_;
    std::vector<Timer> host_timers_;
};

} // namespace vilib
