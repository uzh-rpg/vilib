/*
 * Common GPU feature detector utilities in CUDA
 * detector_base_gpu_cuda_tools.h
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

namespace vilib {

void detector_base_gpu_regular_nms(const int image_width,
                                   const int image_height,
                                   const int horizontal_border,
                                   const int vertical_border,
                                   /* pitch in bytes/sizeof(float) */
                                   const int response_pitch_elements,
                                   float * d_response,
                                   cudaStream_t stream);

void detector_base_gpu_grid_nms(const int image_level,
                                const int min_image_level,
                                const int image_width,
                                const int image_height,
                                const int horizontal_border,
                                const int vertical_border,
                                const int cell_size_width,
                                const int cell_size_height,
                                const int horizontal_cell_num,
                                const int vertical_cell_num,
                                const bool strictly_greater,
                                /* pitch in bytes/sizeof(float) */
                                const int response_pitch_elements,
                                const float * d_response,
                                float2 * d_pos,
                                float * d_score,
                                int * d_level,
                                cudaStream_t stream);

} // namespace vilib
