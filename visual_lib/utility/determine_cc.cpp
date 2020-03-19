/*
 * Standalone utility for determining the CC of the available GPUs
 * determine_cc.cpp
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
