/*
 * Feature tracker utilities in CUDA
 * feature_tracker_cuda_tools.h
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

#include "vilib/common/types.h"

namespace vilib {
namespace feature_tracker_cuda_tools {

void update_tracks(const int candidate_num,
                   const bool affine_est_offset,
                   const bool affine_est_gain,
                   const int min_level,
                   const int max_level,
                   const image_pyramid_descriptor_t & pyramid_description,
                   const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                   const int * d_indir_data,
                   const float2 * d_in_ref_px,
                   unsigned char * d_patch_data,
                   float * d_hessian_data,
                   cudaStream_t stream);

void track_features(const bool affine_est_offset,
                    const bool affine_est_gain,
                    const int candidate_num,
                    const int min_level,
                    const int max_level,
                    const float min_update_squared,
                    const image_pyramid_descriptor_t & pyramid_description,
                    const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                    const int * d_indir_data,
                    const unsigned char * d_patch_data,
                    const float * d_hessian_data,
                    const float2 * d_in_first_px,
                    float2 * d_in_cur_px,
                    float2 * d_in_cur_alpha_beta,
                    float4 * d_in_cur_f,
                    float  * d_in_cur_disparity,
                    cudaStream_t stream);

} // namespace feature_tracker_cuda_tools
} // namespace vilib
