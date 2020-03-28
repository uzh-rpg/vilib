/*
 * Common CUDA functionalities
 * cuda_common.cpp
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

#include <iostream>
#include "vilib/cuda_common.h"

namespace vilib {

bool cuda_initialize(void) {
  /*
   * Note to future self and others:
   * there's no explicit initialization of the CUDA runtime, hence when
   * the first cudaX...() call is made, the CUDA runtime gets initialized.
   * This behaviour can skew benchmark measurements.
   */
  int device_count;
  if(cudaGetDeviceCount(&device_count) != cudaSuccess) {
    std::cout << "Error: no CUDA-capable device detected" << std::endl;
    return false;
  }

  CUDA_API_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  // .. we need to do something more meaningful to initialize the Runtime API..
  unsigned char * temp;
  CUDA_API_CALL(cudaMalloc((void**)&temp,1));
  CUDA_API_CALL(cudaFree(temp));
  return true;
}

kernel_params_t cuda_gen_kernel_params_2d(unsigned int width,
                                          unsigned int height,
                                          unsigned int tpb_x,
                                          unsigned int tpb_y) {
  assert(width > 0 && height > 0);
  kernel_params_t p;
  p.threads_per_block.x = tpb_x;
  p.threads_per_block.y = tpb_y;
  p.threads_per_block.z = 1;
  p.blocks_per_grid.x = (width + tpb_x - 1) / tpb_x;
  p.blocks_per_grid.y = (height + tpb_y - 1) / tpb_y;
  p.blocks_per_grid.z = 1;
  return p;
}

kernel_params_t cuda_gen_kernel_params_1d(unsigned int n,
                                          unsigned int tpb) {
 assert(n > 0);
 kernel_params_t p;
 p.threads_per_block.x = tpb;
 p.threads_per_block.y = 1;
 p.threads_per_block.z = 1;
 p.blocks_per_grid.x = (n + tpb-1)/tpb;
 p.blocks_per_grid.y = 1;
 p.blocks_per_grid.z = 1;
 return p;
}

} // namespace vilib
