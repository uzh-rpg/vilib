# Copyright (c) 2019-2021 Philipp Foehn,
# Robotics and Perception Group, University of Zurich
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.10)
project(vilib VERSION 1.0.0 DESCRIPTION "fast cuda fast feature tracking" LANGUAGES CXX CUDA)

# Packages
find_package(catkin_simple REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Setup catkin simple
catkin_simple()

add_compile_options(-std=c++17)
add_compile_options(-march=native)
add_compile_options(-O3 -DNDEBUG)

set(cxx_options
  -Wall
  -Wextra
  -Werror
  -Wfatal-errors
  -Wno-overloaded-virtual
  -ffast-math
  -fsee
  -fno-signed-zeros
  -fno-math-errno
  -funroll-loops
  -fno-finite-math-only)

set(nvcc_options
  --device-w
  --ftz=true
  --prec-div=false
  --prec-sqrt=false
  --fmad=true
  --default-stream per-thread
  -O3
  -DNDEBUG)

SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
get_filename_component(VILIB_PATH ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(VILIB_PATH "${VILIB_PATH}/visual_lib")
include_directories(${VILIB_PATH}/include)
include_directories(BEFORE SYSTEM ${EIGEN3_INCLUDE_DIRS})

file(GLOB_RECURSE VILIB_SOURCES
  ${VILIB_PATH}/src/*.cpp
  ${VILIB_PATH}/src/common/*.cpp
  ${VILIB_PATH}/src/storage/*.cpp
  ${VILIB_PATH}/src/preprocess/*.cpp
  ${VILIB_PATH}/src/preprocess/*.cu
  ${VILIB_PATH}/src/feature_tracker/*.cpp
  ${VILIB_PATH}/src/feature_tracker/*.cu
  ${VILIB_PATH}/src/feature_detection/*.cpp
  ${VILIB_PATH}/src/feature_detection/*.cu
)

cuda_add_library(vilib SHARED ${VILIB_SOURCES})
target_include_directories(vilib PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if (CMAKE_CUDA_COMPILER)
   target_compile_options(vilib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_options}> $<$<COMPILE_LANGUAGE:CUDA>:${nvcc_options}>)
else()
   target_compile_options(vilib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_options}> $<$<NOT:$<COMPILE_LANGUAGE:CXX>>:${nvcc_options}>)
endif()

target_link_libraries(vilib #PUBLIC
  opencv_core
  opencv_imgproc
  opencv_highgui
  opencv_features2d
  Eigen3::Eigen
  ${CUDA_LIBRARIES}
)

cs_add_executable(vilib_ros
  src/vilib_ros.cpp
  src/vilib_params.cpp
)
target_compile_options(vilib_ros PRIVATE
  -fno-finite-math-only
  -Wall
  -Werror
  -Wpedantic
  -Wunused
  -Wundef
  -Wcast-align
  -Wredundant-decls
  -Wodr
  -Wunreachable-code
  -Wno-overloaded-virtual
)
target_link_libraries(vilib_ros vilib)



# Finish catkin simple
cs_install()
cs_export()

