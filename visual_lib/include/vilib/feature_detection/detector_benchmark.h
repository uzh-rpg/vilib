/*
 * Benchmarking for feature detection tasks
 * detector_benchmark.h
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
