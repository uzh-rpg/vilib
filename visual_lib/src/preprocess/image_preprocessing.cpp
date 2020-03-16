/*
 * High-level functions for image preprocessing
 * image_preprocessing.cpp
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
