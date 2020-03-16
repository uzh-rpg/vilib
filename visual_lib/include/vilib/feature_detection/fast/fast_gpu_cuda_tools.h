/*
 * FAST feature detector utilities in CUDA
 * fast_gpu_cuda_tools.h
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
