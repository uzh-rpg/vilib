/*
 * FAST feature detector utilities in CUDA
 * fast_gpu_cuda_tools.h
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
