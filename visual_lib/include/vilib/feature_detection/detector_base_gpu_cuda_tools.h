/*
 * Common GPU feature detector utilities in CUDA
 * detector_base_gpu_cuda_tools.h
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
                                const bool subpixel_refinement,
                                const bool replace_on_same_level_only,
                                /* pitch in bytes/sizeof(float) */
                                const int response_pitch_elements,
                                const float * d_response,
                                float2 * d_pos,
                                float * d_score,
                                int * d_level,
                                cudaStream_t stream);

} // namespace vilib
