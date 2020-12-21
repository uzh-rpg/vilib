/*
 * Harris/Shi-Tomasi feature detector utilities in CUDA
 * harris_gpu_cuda_tools.h
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

#include "vilib/preprocess/conv_filter.h"

namespace vilib {

void harris_gpu_array_multiply(const float * d_input_a,
                               const int input_a_pitch,
                               const float * d_input_b,
                               const int input_b_pitch,
                               float * d_output,
                               const int output_pitch,
                               const int cols,
                               const int rows,
                               cudaStream_t stream);
void harris_gpu_calc_corner_response(const float * d_dx2,
                                     const int dx2_pitch,
                                     const float * d_dy2,
                                     const int dy2_pitch,
                                     const float * d_dxdy,
                                     const int dxdy_pitch,
                                     float * d_response,
                                     const int response_pitch,
                                     const int cols,
                                     const int rows,
                                     const conv_filter_border_type_t border_type,
                                     const bool use_harris,
                                     const float k,
                                     cudaStream_t stream);

} // namespace vilib
