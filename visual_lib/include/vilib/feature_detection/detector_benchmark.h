/*
 * Benchmarking for feature detection tasks
 * detector_benchmark.h
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
