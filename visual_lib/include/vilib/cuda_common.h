/*
 * Common CUDA functionalities
 * cuda_common.h
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

#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>

namespace vilib {

struct kernel_params {
  dim3 threads_per_block;
  dim3 blocks_per_grid;
};
typedef struct kernel_params kernel_params_t;

#ifndef NDEBUG
#define CUDA_API_CALL(__CALL__) do {                                           \
                                  const cudaError_t a = __CALL__;              \
                                  if (a != cudaSuccess) {                      \
                                    std::cout << "CUDA Error: "                \
                                              << cudaGetErrorString(a)         \
                                              << " (err_num="                  \
                                              << a << ")" << std::endl;        \
                                    std::cout <<  "File: " << __FILE__         \
                                              << " | Line: " << __LINE__       \
                                              << std::endl;                    \
                                    cudaDeviceReset();                         \
                                    assert(0);                                 \
                                  }                                            \
                                } while(0)
#define CUDA_KERNEL_CHECK()     do {                                           \
                                  const cudaError_t a = cudaGetLastError();    \
                                  if (a != cudaSuccess) {                      \
                                    std::cout << "CUDA Error: "                \
                                              << cudaGetErrorString(a)         \
                                              << " (err_num="                  \
                                              << a << ")" << std::endl;        \
                                    std::cout << "File: "                      \
                                              << __FILE__                      \
                                              << " | Line: "                   \
                                              << __LINE__ << std::endl;        \
                                    cudaDeviceReset();                         \
                                    assert(0);                                 \
                                  }                                            \
                                } while(0)
#else
#define CUDA_API_CALL(__CALL__) __CALL__
#define CUDA_KERNEL_CHECK()
#endif /* NDEBUG */

/*
 * Initialize the CUDA Runtime API and query GPU specific parameters
 */
bool cuda_initialize(void);

/*
 * Generate 1D kernel launch parameters
 * @param total_n number of total launched threads
 * @param tpb number of threads per block
 */
kernel_params_t cuda_gen_kernel_params_1d(unsigned int total_n,
                                          unsigned int tpb);

/*
 * Generate 2D kernel launch parameters
 * @param width number of total threads (e.g.: image width) in the horizontal direction
 * @param height number of total threads (e.g.: image height) in the vertical direction
 * @param tpb_x number of threads per block in the horizontal direction
 * @param tpb_y number of threads per block in the vertical direction
 */
kernel_params_t cuda_gen_kernel_params_2d(unsigned int width,
                                          unsigned int height,
                                          unsigned int tpb_x,
                                          unsigned int tpb_y);

} // namespace vilib
