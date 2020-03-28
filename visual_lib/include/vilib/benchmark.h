/*
 * Benchmark file
 * benchmark.h
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
