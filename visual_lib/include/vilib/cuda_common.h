/*
 * Common CUDA functionalities
 * cuda_common.h
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
