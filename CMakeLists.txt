#
# CMakefile for the CUDA visual library (ViLib)
#
# Copyright (c) 2019-2020 Balazs Nagy,
# Robotics and Perception Group, University of Zurich
# Copyright (c) 2021 Bernd Pfrommer
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

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  project(vilib VERSION 1.0.0 DESCRIPTION "fast cuda fast feature tracking" LANGUAGES CXX CUDA)
else()
  message(STATUS "CUDA not natively supported yet, will look for CUDA cmake package")
  project(vilib VERSION 1.0.0 DESCRIPTION "fast cuda fast feature tracking" LANGUAGES CXX)
endif()

include (CMakePrintHelpers)
include (CMakePackageConfigHelpers)
include (CPack)
include (GNUInstallDirs)

if (CMAKE_CUDA_COMPILER)
else()
   find_package(CUDA REQUIRED)
   set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
   cuda_select_nvcc_arch_flags(ARCH_FLAGS)
   message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
endif()


find_package(OpenCV REQUIRED COMPONENTS core)
find_package(Eigen3 REQUIRED)

option(BUILD_SHARED_LIBS "build shared libraries" ON)
cmake_print_variables(BUILD_SHARED_LIBS)


set(CUSTOM_OPENCV_PATH "" CACHE PATH "path to custom opencv version")

file(GLOB_RECURSE VILIB_SOURCES
  visual_lib/src/*.cpp
  visual_lib/src/*.cu
)

if (CMAKE_CUDA_COMPILER)
   add_library(vilib SHARED ${VILIB_SOURCES})
else()
   cuda_add_library(vilib SHARED ${VILIB_SOURCES})
endif()

# the PUBLIC keyword means that any targets linking to this
# target also need that include directory. 
target_include_directories(vilib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/visual_lib/include>  
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    )

# will require that the compiler support compiler standard, and
# add the flag if necessary. The PUBLIC keyword means any targets depending
# on this target will also have to satisfy that requirement
target_compile_features(vilib PUBLIC cxx_std_11)

# tell cmake to use e.g. -std=c++11 instead of -std=g++11, and bomb
# out if the standard is not required
set_target_properties(vilib
  PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS OFF
  LINK_WHAT_YOU_USE OFF         # switch to on to check for unnecessary links
  POSITION_INDEPENDENT_CODE ON
  )

target_link_libraries(vilib PUBLIC
  opencv_core Eigen3::Eigen ${CUDA_LIBRARIES})

# Set different compiler options for cxx and nvcc

set(cxx_options -Wall -Wextra -Werror -Wfatal-errors -ffast-math -fsee -fno-signed-zeros -fno-math-errno -funroll-loops -fno-finite-math-only -march=native -O3 -DNDEBUG)
set(nvcc_options --device-w --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --default-stream per-thread -O3 -DNDEBUG)

# the generator is only expanded if <COMPILE_LANGUAGE:FOO> evaluates to true

if (CMAKE_CUDA_COMPILER)
   target_compile_options(vilib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_options}> $<$<COMPILE_LANGUAGE:CUDA>:${nvcc_options}>)
else()
   target_compile_options(vilib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_options}> $<$<NOT:$<COMPILE_LANGUAGE:CXX>>:${nvcc_options}>)
endif()

# Specify how to install the target.
# The EXPORT keyword associates the installed target files with an "export" called
# ${PROJECT_NAME}Targets. The file will later be generated when later install(EXPORT) is
# called on that target.
# Note that this install does not install the header files yet.
install(TARGETS vilib
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION include
  )

install(DIRECTORY visual_lib/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# generate the ConfigVersion.cmake file that will be included by Config.cmake
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

# generate the Config.cmake file
configure_file(cmake/${PROJECT_NAME}Config.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)

# generate the Targets.cmake file from the export info learned during the
# installation of the vilib target
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# install the generated version and config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        )


# Packaging support
set(CPACK_PACKAGE_VENDOR "vilib")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "fast cuda based FAST feature lib")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENCE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
