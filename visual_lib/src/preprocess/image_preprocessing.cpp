/*
 * High-level functions for image preprocessing
 * image_preprocessing.cpp
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

#include "vilib/preprocess/image_preprocessing.h"
#include "vilib/preprocess/pyramid.h"
#include "vilib/feature_detection/detector_benchmark.h"

namespace vilib {

void preprocess_image(const cv::Mat & img,
                      std::vector<std::shared_ptr<Subframe>> & pyramid,
                      cudaStream_t stream) {
  // Copy input to preallocated buffer
  BENCHMARK_START_HOST(DetectorBenchmark,Upload,true);
  pyramid[0]->copy_from(img,true,stream);
  BENCHMARK_STOP_HOST(DetectorBenchmark,Upload);

  // Create the pyramid
  BENCHMARK_START_DEVICE(DetectorBenchmark,Pyramid,0);
  pyramid_create_gpu(pyramid,stream);
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,Pyramid,0);
}

#ifdef ROS_SUPPORT
void preprocess_image(const sensor_msgs::ImageConstPtr & msg,
                      std::vector<std::shared_ptr<Subframe>> & pyramid,
                      cudaStream_t stream) {
  // Copy input to preallocated buffer
  // Note: it will be only asynchoronous if the source is pinned
  BENCHMARK_START_HOST(DetectorBenchmark,Upload,true);
  pyramid[0]->copy_from(msg,true,stream);
  BENCHMARK_STOP_HOST(DetectorBenchmark,Upload);

  // Create the pyramid
  BENCHMARK_START_DEVICE(DetectorBenchmark,Pyramid,0);
  pyramid_create_gpu(pyramid,stream);
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,Pyramid,0);
}
#endif /* ROS_SUPPORT */

} // namespace vilib
