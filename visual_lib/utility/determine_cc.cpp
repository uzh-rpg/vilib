/*
 * Standalone utility for determining the CC of the available GPUs
 * determine_cc.cpp
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
#include <string>
#include <cuda_runtime_api.h>

#define CUDA_API_CALL(__CALL__)  do {                                           \
                                   const cudaError_t a = __CALL__;              \
                                   if (a != cudaSuccess) {                      \
                                     std::cout << "CUDA Error: " <<             \
                                      cudaGetErrorString(a)   <<                \
                                      "(err_num=" << a << ")" << std::endl;     \
                                     std::cout << "File: " << __FILE__ <<       \
                                       " | Line: " << __LINE__ << std::endl;    \
                                     cudaDeviceReset();                         \
                                     return -1;                                 \
                                   }                                            \
                                 } while(0)

int main(int argc, char * argv[]) {
  int device_count = 0;
  CUDA_API_CALL(cudaGetDeviceCount(&device_count));
  if(device_count == 0) {
    std::cout << "No CUDA-capable device was found" << std::endl;
    return -1;
  }

  std::string nvcc_arg;
  for(int i=0;i<device_count;++i) {
    cudaSetDevice(i);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    if(nvcc_arg.length()) {
      nvcc_arg += " ";
    }
    nvcc_arg += "--generate-code arch=compute_"
             + std::to_string(deviceProp.major)
             + std::to_string(deviceProp.minor)
             + ",code=sm_"
             + std::to_string(deviceProp.major)
             + std::to_string(deviceProp.minor);
  }
  std::cout << nvcc_arg << std::endl;
  return 0;
}
