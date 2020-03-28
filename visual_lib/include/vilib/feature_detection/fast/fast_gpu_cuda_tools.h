/*
 * FAST feature detector utilities in CUDA
 * fast_gpu_cuda_tools.h
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

#include "vilib/feature_detection/fast/fast_common.h"
#include "vilib/feature_detection/fast/fast_gpu_config.h"

namespace vilib {

unsigned char fast_gpu_is_corner_host(const unsigned int & address,
                                      const int & min_arc_length);
void fast_gpu_calculate_lut(
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                            unsigned int * d_corner_lut,
#else
                            unsigned char * d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                            const int & min_arc_length,
                            cudaStream_t stream);
void fast_gpu_calc_corner_response(const int image_width,
                                   const int image_height,
                                   const int image_pitch,
                                   const unsigned char * d_image,
                                   const int horizontal_border,
                                   const int vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                   const unsigned int * d_corner_lut,
#else
                                   const unsigned char * d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                   const float threshold,
                                   const int min_arc_length,
                                   const fast_score score,
                                   /* pitch in bytes/sizeof(float) */
                                   const int response_pitch_elements,
                                   float * d_response,
                                   cudaStream_t stream);

} // namespace vilib
