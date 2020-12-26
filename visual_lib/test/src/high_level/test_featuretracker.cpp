/*
 * Tests for feature tracking functionalities
 * test_featuretracker.cpp
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

#include <unordered_map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "vilib/config.h"
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/cuda_common.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"
#include "test/high_level/test_featuretracker.h"
#include "test/groundtruth/blender.h"
#include "test/groundtruth/gradient_image.h"

using namespace vilib;

// Frame options
#define FRAME_IMAGE_PYRAMID_LEVELS            5
// Feature detection options
#define FEATURE_DETECTOR_CELL_SIZE_WIDTH      32
#define FEATURE_DETECTOR_CELL_SIZE_HEIGHT     32
#define FEATURE_DETECTOR_MIN_LEVEL            0
#define FEATURE_DETECTOR_MAX_LEVEL            2
#define FEATURE_DETECTOR_HORIZONTAL_BORDER    8
#define FEATURE_DETECTOR_VERTICAL_BORDER      8

// Feature detector selection
#define FEATURE_DETECTOR_FAST                 0
#define FEATURE_DETECTOR_HARRIS               1
#define FEATURE_DETECTOR_SHI_TOMASI           2
#define FEATURE_DETECTOR_USED                 FEATURE_DETECTOR_HARRIS

// FAST parameters
#define FEATURE_DETECTOR_FAST_EPISLON         10.f
#define FEATURE_DETECTOR_FAST_ARC_LENGTH      10
#define FEATURE_DETECTOR_FAST_SCORE           SUM_OF_ABS_DIFF_ON_ARC
// Harris/Shi-Tomasi parameters
#define FEATURE_DETECTOR_HARRIS_K             0.04f
#define FEATURE_DETECTOR_HARRIS_QUALITY_LEVEL 0.1f
#define FEATURE_DETECTOR_HARRIS_BORDER_TYPE   conv_filter_border_type::BORDER_SKIP

// Test framework options
#define START_AT_FRAME_ID_N                   0
#define TERMINATE_AFTER_FRAME_ID_N            -1
#define VISUALIZE_FEATURE_TRACKING            0
#define SAVE_INSTEAD_DISPLAY                  0
// Test framework statistics
#define STAT_ID_TRACKER_TIMER                 0
#define STAT_ID_TRACKED_FEATURE_COUNT         1
#define STAT_ID_DETECTED_FEATURE_COUNT        2
#define STAT_ID_TOTAL_FEATURE_COUNT           3
#define STAT_ID_RMSE_PER_FEATURE              4
#define STAT_ID_RMSE_PER_FRAME                5

TestFeatureTracker::TestFeatureTracker(const char * file_path, const int max_image_num) :
  TestBase("Feature Tracker",file_path,max_image_num),initialized_(false) {
}

bool TestFeatureTracker::run(void) {
  if(!initialized_) {
    // Instantiation of the trackers
    FeatureTrackerOptions feature_tracker_options;
    feature_tracker_options.reset_before_detection = false;
    feature_tracker_options.use_best_n_features = 50;
    feature_tracker_options.min_tracks_to_detect_new_features = 0.3*feature_tracker_options.use_best_n_features;
    feature_tracker_options.affine_est_gain = false;
    feature_tracker_options.affine_est_offset = false;

    // Instantiate detectors (default is the Blender dataset)
    if(!load_image_dimensions(752,480)) {
      // Could not acquire the initialization parameters
      return false;
    }
    // Create feature detector & tracker for the GPU
#if (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_FAST)
    detector_gpu_.reset(new FASTGPU(image_width_,
                                    image_height_,
                                    FEATURE_DETECTOR_CELL_SIZE_WIDTH,
                                    FEATURE_DETECTOR_CELL_SIZE_HEIGHT,
                                    FEATURE_DETECTOR_MIN_LEVEL,
                                    FEATURE_DETECTOR_MAX_LEVEL,
                                    FEATURE_DETECTOR_HORIZONTAL_BORDER,
                                    FEATURE_DETECTOR_VERTICAL_BORDER,
                                    FEATURE_DETECTOR_FAST_EPISLON,
                                    FEATURE_DETECTOR_FAST_ARC_LENGTH,
                                    FEATURE_DETECTOR_FAST_SCORE));
#elif (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_HARRIS || FEATURE_DETECTOR_USED == FEATURE_DETECTOR_SHI_TOMASI)
    detector_gpu_.reset(new HarrisGPU(image_width_,
                                      image_height_,
                                      FEATURE_DETECTOR_CELL_SIZE_WIDTH,
                                      FEATURE_DETECTOR_CELL_SIZE_HEIGHT,
                                      FEATURE_DETECTOR_MIN_LEVEL,
                                      FEATURE_DETECTOR_MAX_LEVEL,
                                      FEATURE_DETECTOR_HORIZONTAL_BORDER,
                                      FEATURE_DETECTOR_VERTICAL_BORDER,
                                      FEATURE_DETECTOR_HARRIS_BORDER_TYPE,
                                      (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_HARRIS),
                                      FEATURE_DETECTOR_HARRIS_K,
                                      FEATURE_DETECTOR_HARRIS_QUALITY_LEVEL));
#endif /* FEATURE_DETECTOR_USED */
    tracker_gpu_.reset(new FeatureTrackerGPU(feature_tracker_options,1));
    tracker_gpu_->setDetectorGPU(detector_gpu_,0);

    // Initialize the pyramid pool
    PyramidPool::init(1,
                      image_width_,
                      image_height_,
                      1,  // grayscale
                      FRAME_IMAGE_PYRAMID_LEVELS,
                      IMAGE_PYRAMID_MEMORY_TYPE);
    std::cout << " Note: No verification performed" << std::endl;
    initialized_ = true;
  }

  // Create common statistics
  std::vector<Statistics> stat_cpu;
  std::vector<Statistics> stat_gpu;
  stat_gpu.emplace_back("[usec]","Tracker execution time");
  stat_gpu.emplace_back("[1]","Tracked feature count");
  stat_gpu.emplace_back("[1]","Detected feature count");
  stat_gpu.emplace_back("[1]","Total feature count");

  bool success;
  if(is_list_) {
    // Run benchmark suite (it will call run_benchmark()) for us
    success = run_benchmark_suite(stat_cpu,stat_gpu);
  } else {
    // add additional statistics
    stat_gpu.emplace_back("[pixel]","Tracking 2D RMSE per feature");
    stat_gpu.emplace_back("[pixel]","Tracking 2D RMSE per frame");
    success = run_blender(stat_cpu,stat_gpu);
  }

#if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
  tracker_gpu_->showAdditionalStat(true);
#endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */

  // Reset trackers back to their original state
  tracker_gpu_->reset();

  // Deinitialize the pyramid pool (for consecutive tests)
  PyramidPool::deinit();

  return success;
}

void TestFeatureTracker::setTrackerOptions(int use_best_n, int min_tracks_to_detect) {
  tracker_gpu_->setBestNFeatures(use_best_n);
  tracker_gpu_->setMinTracksToDetect(min_tracks_to_detect);
}

bool TestFeatureTracker::run_benchmark(std::vector<Statistics> & stat_cpu,
                                       std::vector<Statistics> & stat_gpu) {
  static int frame_id = 0;
  ++frame_id;
  // Unused
  (void)stat_cpu;

  Timer timer;
  std::size_t total_tracked_ftr_cnt,total_detected_ftr_cnt;
  timer.start();
  // Create Frame
  std::shared_ptr<Frame> frame = std::make_shared<Frame>(
                                              image_,
                                              0,
                                              FRAME_IMAGE_PYRAMID_LEVELS);
  // Create FrameBundle
  std::vector<std::shared_ptr<Frame>> framelist;
  framelist.push_back(frame);
  std::shared_ptr<FrameBundle> framebundle(new FrameBundle(framelist));
  tracker_gpu_->track(framebundle,
                      total_tracked_ftr_cnt,
                      total_detected_ftr_cnt);
  timer.stop();
  stat_gpu.at(STAT_ID_TRACKER_TIMER).add(timer.elapsed_usec());
  stat_gpu.at(STAT_ID_TRACKED_FEATURE_COUNT).add(total_tracked_ftr_cnt);
  stat_gpu.at(STAT_ID_DETECTED_FEATURE_COUNT).add(total_detected_ftr_cnt);
  stat_gpu.at(STAT_ID_TOTAL_FEATURE_COUNT).add(total_tracked_ftr_cnt + total_detected_ftr_cnt);

#if VISUALIZE_FEATURE_TRACKING
  cv::Mat color;
  cv::cvtColor(image_, color, cv::COLOR_GRAY2RGB);
  visualize_features(frame,color,true);
#if SAVE_INSTEAD_DISPLAY
  char frame_filename[20];
  sprintf(frame_filename,"frame_gpu_%04d.png",frame_id);
  cv::imwrite(frame_filename,color);
#else
  display_image(color,"Tracked features (GPU)");
#endif /* SAVE_INSTEAD_DISPLAY */
#endif /* VISUALIZE_FEATURE_TRACKING */
  return true;
}

bool TestFeatureTracker::run_blender(std::vector<Statistics> & stat_cpu,
                                     std::vector<Statistics> & stat_gpu) {
  // Unused
  (void)stat_cpu;

  /*
   * Test scenario:
   * Take the Blender images, & start playing back a sequence:
   * 01) Create frames from the images, which internally initializes the pyramid
   * 02) Pass the frames to the feature tracker:
   *     - initialize the tracker with features to track (Feature detection)
   *     - if not first frame, try to track with the pyramidical inverse-compositional Lukas-Kanade
   * 03) Output and check:
   *     - for every new feature that just starts tracking, get the position of the point
   *       so that it can be reprojected (RMSE = 0 for these points)
   *     - for every tracked point, calculate the reprojection's RMSE, given the camera's
   *       current pose, and the point that was initially extracted
   *     - sum the RMSE-s together, also count the number of points being tracked
   *     - final output: total tracking RMSE, average tracking RMSE, number of points being tracked
   */
  Blender bldr(Blender::Scenery::HutLong);
  const Eigen::MatrixXd M = bldr.get_M();
  std::size_t frame_count = bldr.getFrameCount();

  cv::Mat image;
  Timer timer;
#if VISUALIZE_FEATURE_TRACKING
  cv::Mat display;
#endif /* VISUALIZE_FEATURE_TRACKING */
  for(std::size_t fid=START_AT_FRAME_ID_N;fid<frame_count && fid<((std::size_t)TERMINATE_AFTER_FRAME_ID_N);++fid) {
    const struct Blender::BlenderFrame & bf = bldr.getFrame(fid);
    // load 2D image for tracking
    load_image_to(bf.image_2d_path_.c_str(),cv::IMREAD_GRAYSCALE,image);
    // load depth image for evaluation
    DepthImage depth_image(bf.image_depth_path_.c_str(),
                            bldr.getIntrinsicParameters(),
                            bf.T_W_C_);

    // create the frame from the image
    // now the CPU version's frame creation time is not valid
    // - because I'm still using the GPU and copying back the images
    // Create Frame
    timer.start();
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(
                                            image,
                                            0,
                                            FRAME_IMAGE_PYRAMID_LEVELS);
    // Create FrameBundle
    std::vector<std::shared_ptr<Frame>> framelist{frame};
    std::shared_ptr<FrameBundle> framebundle(new FrameBundle(framelist));
  
    // do the tracking
    std::size_t tracked_feature_cnt, detected_feature_cnt;
    tracker_gpu_->track(framebundle,tracked_feature_cnt,detected_feature_cnt);
    timer.stop();
    stat_gpu[STAT_ID_TRACKER_TIMER].add(timer.elapsed_usec());
    stat_gpu[STAT_ID_TRACKED_FEATURE_COUNT].add(tracked_feature_cnt);
    stat_gpu[STAT_ID_DETECTED_FEATURE_COUNT].add(detected_feature_cnt);
    stat_gpu[STAT_ID_TOTAL_FEATURE_COUNT].add(tracked_feature_cnt + detected_feature_cnt);

    // calculate the proposed error from frame 1 ON
    Eigen::MatrixXd M_current = M * bf.T_C_W_;
    calculate_feature_error(M_current,
                            depth_image,
                            frame,
                            stat_gpu[STAT_ID_RMSE_PER_FEATURE],
                            stat_gpu[STAT_ID_RMSE_PER_FRAME]);

    // visualize the features tracked
#if VISUALIZE_FEATURE_TRACKING
    load_image_to(bf.image_2d_path_.c_str(),cv::IMREAD_COLOR,display);
    visualize_features(frame,display,true);
#if SAVE_INSTEAD_DISPLAY
    static int frame_id = 1;
    ++frame_id;
    char frame_filename[20];
    sprintf(frame_filename,"frame_gpu_%04d.png",frame_id);
    cv::imwrite(frame_filename,display);
#else
    display_image(display,"Feature tracker (GPU)");
#endif /* SAVE_INSTEAD_DISPLAY */
#endif /* VISUALIZE_FEATURE_TRACKING */
  }

  // Display statistics
  std::cout << " GPU ---" << std::endl;
  for(const Statistics & stat : stat_gpu) {
    stat.display();
  }
  return true;
}

void TestFeatureTracker::calculate_feature_error(
      const Eigen::MatrixXd & M,
      const DepthImage & depth_image,
      const std::shared_ptr<Frame> & frame,
      Statistics & rmse_per_feature,
      Statistics & rmse_per_frame) {
  // note: id-s start from 0
  static int last_track_id = -1;
  static std::unordered_map<std::size_t,
                            Eigen::Vector4d,
                            std::hash<std::size_t>,
                            std::equal_to<std::size_t>,
                            Eigen::aligned_allocator<std::pair<const std::size_t,Eigen::Vector4d>>> track_origin_point;
  double tracked_points_rmse = 0.0;
  std::size_t tracked_points = 0;
  for(std::size_t i=0;i<frame->num_features_;++i) {
    // only evaluate error if track id is smaller or equal to last_track_id
    // as we are already know the location of this point
    const Eigen::Vector2d & pos_2d = frame->px_vec_.col(i);
    const int & track_id = frame->track_id_vec_[i];
    if(last_track_id < track_id) {
      // get the 3D position of point from the depth map in world coordinates
      track_origin_point[track_id] = depth_image.point_at((float)pos_2d[0],(float)pos_2d[1]);
    } else {
      // project the world point onto the current canvas, and calculate the error
      Eigen::Vector4d & P_w = track_origin_point[track_id];
      Eigen::Vector3d p = M * P_w;
      p /= p(2);
      double error = std::sqrt(std::pow((p(0)-pos_2d[0]),2) +
                               std::pow((p(1)-pos_2d[1]),2));
      rmse_per_feature.add(error);
      tracked_points_rmse += error;
      ++tracked_points;
    }
  }
  // update the highest track id
  if(frame->num_features_ > 0 && frame->track_id_vec_[frame->num_features_-1] > last_track_id) {
    last_track_id = frame->track_id_vec_[frame->num_features_-1];
  }
  if(tracked_points > 0) {
    rmse_per_frame.add(tracked_points_rmse);
  }
}

void TestFeatureTracker::visualize_features(const std::shared_ptr<Frame> & frame,
                                            cv::Mat & display,
                                            bool draw_cells) {
  static int last_track_id = -1;
  static std::unordered_map<std::size_t,cv::Scalar> track_colors;

  // drawing cells
  if(draw_cells) {
    std::size_t n_rows = (display.rows + FEATURE_DETECTOR_CELL_SIZE_HEIGHT-1)/FEATURE_DETECTOR_CELL_SIZE_HEIGHT;
    std::size_t n_cols = (display.cols + FEATURE_DETECTOR_CELL_SIZE_WIDTH -1)/FEATURE_DETECTOR_CELL_SIZE_WIDTH;
    for(std::size_t r=0;r<n_rows;++r) {
        for(std::size_t c=0;c<n_cols;++c) {
          cv::rectangle(display,
                        cv::Point(c*FEATURE_DETECTOR_CELL_SIZE_WIDTH,r*FEATURE_DETECTOR_CELL_SIZE_HEIGHT),
                        cv::Point((c+1)*FEATURE_DETECTOR_CELL_SIZE_WIDTH,(r+1)*FEATURE_DETECTOR_CELL_SIZE_HEIGHT),
                        cv::Scalar(244,215,66), // B,G,R
                        1,
                        8,
                        0);
        }
      }
  }

  // note: id-s start from 0
  for(std::size_t i=0;i<frame->num_features_;++i) {
    const int SHIFT_BITS = 10;
    // Circle center
    const Eigen::Vector2d & pos_2d = frame->px_vec_.col(i);
    float x = pos_2d[0] * (1<<SHIFT_BITS);
    float y = pos_2d[1] * (1<<SHIFT_BITS);
    // Track id
    const int & track_id = frame->track_id_vec_[i];
    // Color: B,G,R
    cv::Scalar track_color(255,255,255);
    if(last_track_id < track_id) {
      // new feature: generate random color, but dont use is yet
      int channel_b = rand() % 255;
      int channel_g = rand() % 255;
      int channel_r = rand() % 255;
      track_colors[(std::size_t)track_id]=cv::Scalar(channel_b,channel_g,channel_r);
    } else {
      // tracked feature: use old color
      track_color = track_colors[track_id];
    }
    cv::circle(display,
               cv::Point((int)x,(int)y),
               1*3*(1<<SHIFT_BITS),
               track_color,
               3, // thickness
               8, // line type
               SHIFT_BITS // shift: number of fractional bits in the coordinates AND the radius
               );
  }
  // update the highest track id
  if(frame->num_features_ > 0 && frame->track_id_vec_[frame->num_features_-1] > last_track_id) {
    last_track_id = frame->track_id_vec_[frame->num_features_-1];
  }
}
