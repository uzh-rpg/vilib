/*
 * Common GPU feature detector utilities in CUDA
 * detector_base_gpu_cuda_tools.h
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
                                /* pitch in bytes/sizeof(float) */
                                const int response_pitch_elements,
                                const float * d_response,
                                float2 * d_pos,
                                float * d_score,
                                int * d_level,
                                cudaStream_t stream);

} // namespace vilib
