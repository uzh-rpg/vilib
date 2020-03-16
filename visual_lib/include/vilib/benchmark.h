/*
 * Benchmark file
 * benchmark.h
 */

#pragma once

namespace vilib {

#define BENCHMARK_ENABLE            	0

#if BENCHMARK_ENABLE
#define BENCHMARK_START_HOST(__CLASS__,__TYPE__,__SYNC__)               __CLASS__::startHost(__CLASS__::Host::__TYPE__,__SYNC__)
#define BENCHMARK_STOP_HOST(__CLASS__,__TYPE__)                         __CLASS__::stopHost(__CLASS__::Host::__TYPE__)
#define BENCHMARK_START_DEVICE(__CLASS__,__TYPE__,__STREAM__)           __CLASS__::startDevice(__CLASS__::Device::__TYPE__,__STREAM__)
#define BENCHMARK_STOP_DEVICE(__CLASS__,__TYPE__,__STREAM__)            __CLASS__::stopDevice(__CLASS__::Device::__TYPE__,__STREAM__)
#define BENCHMARK_COLLECT_DEVICE(__CLASS__,__TYPE__)                    __CLASS__::collectDevice(__CLASS__::Device::__TYPE__)
#else
#define BENCHMARK_START_HOST(__CLASS__,__TYPE__,__SYNC__)
#define BENCHMARK_STOP_HOST(__CLASS__,__TYPE__)
#define BENCHMARK_START_DEVICE(__CLASS__,__TYPE__,__STREAM__)
#define BENCHMARK_COLLECT_DEVICE(__CLASS__,__TYPE__)
#define BENCHMARK_STOP_DEVICE(__CLASS__,__TYPE__,__STREAM__)
#endif /* BENCHMARK_ENABLE */

} // namespace vilib
