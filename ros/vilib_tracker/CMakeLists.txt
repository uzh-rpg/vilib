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
project(vilib_tracker
  VERSION 1.0.0
  DESCRIPTION "Fast CUDA Feature Tracker"
  LANGUAGES CXX
)

# Package Dependencies
find_package(catkin REQUIRED
  COMPONENTS
  vilib_msgs
  vilib_catkin
  cv_bridge
  image_transport
  roscpp
)

# Catkin Package Setup
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  vilib_msgs
  vilib_catkin
  cv_bridge
  image_transport
  roscpp
)

# Compile Flags
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  message(STATUS "Using ARMv7 optimized flags!")
  set(VILIB_ARCH_FLAGS "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
  add_definitions(-D__ARM_NEON__)
elseif ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  message(STATUS "Using ARM aarch64 optimized flags!")
  set(VILIB_ARCH_FLAGS "-march=native")
  add_definitions(-D__ARM_NEON__)
else ()
  set(VILIB_ARCH_FLAGS "-march=native")
endif ()

set(VILIB_CXX_FLAGS
  ${VILIB_ARCH_FLAGS}
  -std=c++17
  -O3
  -DNDEBUG
  -DROS_SUPPORT
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
  -fno-finite-math-only
)

add_executable(${PROJECT_NAME}
  src/vilib_ros.cpp
  src/vilib_params.cpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${VILIB_CXX_FLAGS}>
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
