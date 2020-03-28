/*
 * FAST feature detector configuration
 * fast_gpu_config.h
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

/*
 * Note to future self:
 * Lessons learnt from implementing FAST on GPU:
 * - recomputing offsets is faster then getting it from constant memory
 * - constant memory serializes accesses if not the same address is accessed within warp
 *   -> constant memory is not a good choice for lookup table
 * - shared memory: this is sad, but the amount of data reuse is not enough to overcome the shm loading/synchronization overheads
 *   after profiling, the arithmetic instructions (due to the complicated SHM loading,
 *   increased the kernel time :( -> you can still find the shared memory version in the commits before "GPU feature detector overhaul"
 * - precomputing the pitches help!
 * - literals are better than const memory accesses
 * - offset computing locally is better than constant memory (e.g. for NMS it was -20usec)
 * - precheck -10 usec (coarse corner response function)
 * - compute response only for the points where the bresenham circle passed the test (-13usec)
 * - texture memory helped during response calculation (-10usec, because the borders were not calculated anymore), but NMS got considerably worse with texture memory
 * - integer multiply is expensive, use the tenary operator if possible
 */
/*
 * Lookup table
 * Note to future self:
 * we could also do some bit magic with __clz().
 * Downside of a lookup table is the serialized accesses to global memory
 * -50usec from kernel runtime
 */
#define FAST_GPU_USE_LOOKUP_TABLE              1
/*
 * Lookup table bit-based
 * Note to future self:
 * bitbase lookup table:
 * lower 11 bit -> address of the 4 bytes
 * upper 5  bit (32 bits) -> we have a 1 where the combination is good
 * ok, sadly it did not help significantly
 */
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED     1

} // namespace vilib
