#
# Makefile for the CUDA visual library (ViLib)
# 
# Copyright (c) 2019-2020 Balazs Nagy,
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

# Options
# Release/Debug mode
RELEASE_MODE=1
#
# Add specific compilation flags for better CPU profiling experience at the cost
# of some performance penalty (for better back-tracing)
CPU_PROFILING_MODE=0
#
# Add specific compilation flags for better GPU Profiling experience at the cost
# of some performance penalty (for line info within visual profiler)
GPU_PROFILING_MODE=0
#
# Display kernel information during compilation
# - register count per kernel
# - spilled stores and loads
GPU_KERNEL_INFO=1

#
# Dependencies
#
# CUDA installation path (mandatory)
CUDA_INSTALLATION_PATH=/usr/local/cuda
#
# Eigen installation path (mandatory)
EIGEN_INSTALLATION_PATH=/usr/include/eigen3
#
# Custom OpenCV support (optional)
# Note: if 0 it will use the OpenCV g++ finds by default
CUSTOM_OPENCV_SUPPORT=0
CUSTOM_OPENCV_INSTALLATION_PATH=/usr/local/opencv
#
# ROS support (optional)
# Note: the release will be automatically selected
ROS_SUPPORT?=0

#
# Output names
SHARED_LIBRARY_NAME=libvilib.so
TEST_SUITE_NAME=test_vilib
CC_UTILITY=determine_cc
#
# Output library installation paths
LIBRARY_INSTALLATION_PATH=/usr/local/vilib

#
# Compilers & binary utilities
# NVCC
#
# Compile each .cu file into an object file that contains executable device code:
# --device-w
#
# Single-precision denormal support:
# --ftz=true  -> flushes denormal values to zero
# --ftz=false -> preserves denormal values
#
# Single-precision floating-point divison and reciprocals:
# --prec-div=true  -> enables the IEEE round-to-nearest mode
# --prec-div=false -> enables the fast approximation mode
#
# Single-precision floating-point square root:
# --prec-sqrt=true  -> enables the IEEE round-to-nearest mode
# --prec-sqrt=false -> enables the fast approximation mode
#
# Single-precision fused multiply and accumulate
# --fmad=true  -> enable fused multiply and accumulate instructions
# --fmad=false -> disable ~
#
# Enable per-thread default streams instead of using the legacy stream 0
# More on this:
# https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
# --default-stream per-thread
#
NVCC=nvcc
NVCC_FLAGS=-std=c++11 --device-w \
           --ftz=true --prec-div=false --prec-sqrt=false --fmad=true \
           --default-stream per-thread

# CXX
CXX=g++
CXX_FLAGS=-std=c++11 -Wall -Wextra -Werror -Wfatal-errors      		 \
          -ffast-math -fsee -fno-signed-zeros                        \
          -fno-math-errno -funroll-loops -fno-finite-math-only		 \
          -march=native
CXX_DEFINES=
CXX_LD_FLAGS=
CXX_LD_DIRS=-L${CUDA_INSTALLATION_PATH}/lib64
CXX_LD_LIBRARIES=-lcudart
# Common
CXX_NVCC_INCLUDES=-Iinclude/ \
                  -isystem ${CUDA_INSTALLATION_PATH}/include \
                  -isystem ${EIGEN_INSTALLATION_PATH}

#
# Build sources & outputs
BUILD_DIR=build
CXX_SOURCES=$(shell find src -type f -name *.cpp)
NVCC_SOURCES=$(shell find src -type f -name *.cu)

# Generate the test suite in addition to the library
ifeq ($(MAKECMDGOALS),test)
CXX_SOURCES+=$(shell find test/src -type f -name *.cpp)
CXX_NVCC_INCLUDES+=-Itest/include

# Generate Source-in-PTX files
else ifeq ($(MAKECMDGOALS),ptx)
NVCC_FLAGS+=-ptx -src-in-ptx --generate-line-info
NVCC_PTX_FILES=$(NVCC_SOURCES:%.cu=${BUILD_DIR}/%.ptx)
endif

NVCC_OBJECTS=$(NVCC_SOURCES:%.cu=${BUILD_DIR}/%.o)
CXX_OBJECTS=$(CXX_SOURCES:%.cpp=${BUILD_DIR}/%.o)
OBJECTS=$(NVCC_OBJECTS) $(CXX_OBJECTS)
DEPENDENCIES=$(OBJECTS:.o=.d)
DIRECTORIES=$(sort $(dir $(OBJECTS)))

# Custom OpenCV support
ifeq (${CUSTOM_OPENCV_SUPPORT},1)
CXX_NVCC_INCLUDES+=-I${CUSTOM_OPENCV_INSTALLATION_PATH}/include          \
                   -I${CUSTOM_OPENCV_INSTALLATION_PATH}/include/opencv4
CXX_LD_DIRS+=-L${CUSTOM_OPENCV_INSTALLATION_PATH}/lib
CXX_LD_FLAGS+=-Wl,-rpath,${CUSTOM_OPENCV_INSTALLATION_PATH}/lib
CXX_LD_LIBRARIES+=-lopencv_imgproc -lopencv_highgui -lopencv_core  \
                  -lopencv_imgcodecs -lopencv_features2d
else
CXX_LD_LIBRARIES+=$(shell pkg-config --libs-only-l opencv)
endif

# ROS support
ifeq (${ROS_SUPPORT},1)
CXX_DEFINES+=-DROS_SUPPORT
CXX_NVCC_INCLUDES+=$(shell pkg-config --cflags rosbag)
CXX_LD_DIRS+=$(shell pkg-config --libs-only-L rosbag)
CXX_LD_LIBRARIES+=$(shell pkg-config --libs-only-l rosbag) -lboost_system
endif

# CPU architecture specific options
CPU_ARCHITECTURE=$(shell uname -p)
ifneq (,$(findstring arm,$(CPU_ARCHITECTURE)))
	CXX_FLAGS+=-mfloat-abi=hard -mfpu=neon
	CXX_DEFINES+=-D__ARM_NEON__
else ifneq (,$(findstring aarch64,$(CPU_ARCHITECTURE)))
	CXX_DEFINES+=-D__ARM_NEON__
endif

# Release/Debug mode
# Note: NDEBUG disables asserts AND the CUDA API call checks in cuda_common.h
ifeq ($(RELEASE_MODE),1)
NVCC_FLAGS+=-O3 -DNDEBUG
CXX_FLAGS+=-O3 -DNDEBUG
else
NVCC_FLAGS+=-O0 -g -G
CXX_FLAGS+=-O0 -g
endif

# Profiling mode
ifeq ($(CPU_PROFILING_MODE),1)
CXX_FLAGS+=-fno-omit-frame-pointer -funwind-tables -g
else
CXX_FLAGS+=-fomit-frame-pointer
endif

ifeq ($(GPU_PROFILING_MODE),1)
NVCC_FLAGS+=-lineinfo
endif

# Display kernel information during compilation
ifeq ($(GPU_KERNEL_INFO),1)
NVCC_FLAGS+=--ptxas-options=-v
endif

default: help

# Primary targets
solib: ${SHARED_LIBRARY_NAME}
test: ${TEST_SUITE_NAME}
ptx: ${NVCC_PTX_FILES}

install:
	mkdir -p ${LIBRARY_INSTALLATION_PATH}/lib
	mkdir -p ${LIBRARY_INSTALLATION_PATH}/include
	cp ${BUILD_DIR}/${SHARED_LIBRARY_NAME} ${LIBRARY_INSTALLATION_PATH}/lib/.
	cp -r include/vilib ${LIBRARY_INSTALLATION_PATH}/include/.

uninstall:
	rm -rf ${LIBRARY_INSTALLATION_PATH}

${SHARED_LIBRARY_NAME}: ${OBJECTS}
	${CXX} ${CXX_LD_FLAGS} -shared ${CXX_LD_DIRS} $^ ${CXX_LD_LIBRARIES} -o ${BUILD_DIR}/$@

${TEST_SUITE_NAME}: ${OBJECTS}
	${CXX} ${CXX_LD_FLAGS} ${CXX_LD_DIRS} $^ ${CXX_LD_LIBRARIES} -o $@

# Note to future self: compile the library .cpp-s with -fPIC, the test environment normally
${BUILD_DIR}/src/%.o: src/%.cpp
	${CXX} ${CXX_FLAGS} -fPIC -MMD ${CXX_DEFINES} ${CXX_NVCC_INCLUDES} -c $< -o $@

${BUILD_DIR}/test/%.o: test/%.cpp
	${CXX} ${CXX_FLAGS} -MMD ${CXX_DEFINES} ${CXX_NVCC_INCLUDES} -c $< -o $@

${BUILD_DIR}/%.ptx: %.cu
	${NVCC} ${NVCC_FLAGS} ${NVCC_ARCHS} ${CXX_NVCC_INCLUDES} -Xcompiler "${CXX_FLAGS} -fPIC ${CXX_DEFINES}" $< -o $@


# Note to future self: the nvcc compiler on the Jetson does not support -MM and
#                      -MF flags
${BUILD_DIR}/%.o: %.cu
	${NVCC} -std=c++11 ${CXX_NVCC_INCLUDES} ${CXX_DEFINES} -M -MT $@ $< -o $(@:.o=.d)
	${NVCC} ${NVCC_FLAGS} ${NVCC_ARCHS} ${CXX_NVCC_INCLUDES} -Xcompiler "${CXX_FLAGS} -fPIC ${CXX_DEFINES}" $< -o $@

${OBJECTS} ${NVCC_PTX_FILES}: | init_dirs init_cc

init_dirs:
	@mkdir -p ${DIRECTORIES}

init_cc:
	$(eval NVCC_ARCHS=$(shell ${NVCC} -std=c++11 utility/determine_cc.cpp -o ${CC_UTILITY}; ./${CC_UTILITY}; rm -rf ${CC_UTILITY}))

help:
	@echo "Available targets:"
	@echo "solib\t\t- Compile a shared library"
	@echo "ptx\t\t- Generate Source-in-PTX files for kernel evaluation"
	@echo "install\t- nstall only the shared library & the headers"
	@echo "uninstall\t- Uninstall the static/shared library & headers"
	@echo "test\t\t- Compile the test suite for the implemented functionalities"

clean:
	rm -rf ${BUILD_DIR}
	rm -rf ${CC_UTILITY}
	rm -rf ${TEST_SUITE_NAME}

.PHONY: clean help init_dirs init_cc

# Include object dependencies
-include $(DEPENDENCIES)
