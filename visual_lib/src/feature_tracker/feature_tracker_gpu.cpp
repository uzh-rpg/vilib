/*
 * Feature tracker GPU
 * feature_tracker_gpu.cpp
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

#include <numeric>
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib/feature_tracker/feature_tracker_options.h"
#include "vilib/feature_tracker/feature_tracker_cuda_tools.h"
#include "vilib/common/point.h"
#include "vilib/cuda_common.h"

namespace vilib {

#define VERBOSE_TRACKING                      0
/*
 * Note to future self:
 * we rely on this byte size in various parts of the code
 */
#define METADATA_ELEMENT_BYTES                64

FeatureTrackerGPU::FeatureTrackerGPU(const FeatureTrackerOptions & options,
                                     const std::size_t & camera_num) :
  FeatureTrackerBase(options,camera_num),
  pyramid_levels_(options_.klt_max_level - options_.klt_min_level + 1) {
  assert(options_.klt_max_level > options_.klt_min_level &&
         "Maximum level should exceed minimum level");
  // Buffer pointers
  buffer_.resize(camera_num);
  for(std::size_t c=0;c<camera_num; ++c) {
    buffer_[c].h_indir_data_ = nullptr;
    buffer_[c].h_metadata_   = nullptr;
    buffer_[c].d_patch_data_ = nullptr;
    buffer_[c].d_hessian_data_ = nullptr;
  }
  // Detectors, streams
  detector_.resize(camera_num,nullptr);
  stream_.resize(camera_num,nullptr);
  // Search for maximum patch size, including borders
  int max_patch_size = 0;
  for(std::size_t i=0;i<options_.klt_patch_sizes.size();++i) {
    pyramid_patch_sizes_.wh[i] = options_.klt_patch_sizes[i];
    if(options_.klt_patch_sizes[i] > max_patch_size) {
      max_patch_size = options_.klt_patch_sizes[i];
    }
  }
  pyramid_patch_sizes_.max_area = (max_patch_size+2)*(max_patch_size+2);
}

FeatureTrackerGPU::~FeatureTrackerGPU(void) {
  for(std::size_t c=0;c<buffer_.size();++c) {
    freeStorage(c);
  }
}

void FeatureTrackerGPU::track(const std::shared_ptr<FrameBundle> & cur_frames,
                              std::size_t & total_tracked_features_num,
                              std::size_t & total_detected_features_num) {
  assert(cur_frames->size() == detector_.size() &&
         "The frame count and the detector count differs");
#if VERBOSE_TRACKING
  static std::size_t frame_bundle_id = 1;
  std::cout << "Frame bundle " << (frame_bundle_id++) << " -------------" << std::endl;
#endif /* VERBOSE_TRACKING */

  std::vector<image_pyramid_descriptor_t> cur_pyramids;
  std::vector<std::shared_ptr<Frame>> cur_base_frames;

  // 00) Prerequisites
  for(std::size_t c=0;c<cur_frames->size();++c) {
    assert(detector_[c] != nullptr && "One must set a GPU feature detector first");
    cur_base_frames.push_back(cur_frames->at(c));
    cur_pyramids.push_back(cur_base_frames[c]->getPyramidDescriptor());
    cur_base_frames[c]->resizeFeatureStorage(max_ftr_count_);
  }

  // 01) Track existing feature tracks (only if there is any)
  for(std::size_t c=0;c<cur_frames->size();++c) {
    if(tracks_[c].size() > 0) {
      for(std::size_t track_id=0; track_id<tracks_[c].size(); ++track_id) {
        struct FeatureTrack & track = tracks_[c][track_id];
        // opposed to the previous version, now we only need to update
        // the indirection layer that points to the already occupied buffer cells
        buffer_[c].h_indir_data_[track_id] = track.buffer_id_;
      }

      // run the tracking on the GPU
      feature_tracker_cuda_tools::track_features(options_.affine_est_offset,
                                                 options_.affine_est_gain,
                                                 tracks_[c].size(),
                                                 options_.klt_min_level,
                                                 options_.klt_max_level,
                                                 options_.klt_min_update_squared,
                                                 cur_pyramids[c],
                                                 pyramid_patch_sizes_,
                                                 buffer_[c].d_indir_data_,
                                                 buffer_[c].d_patch_data_,
                                                 buffer_[c].d_hessian_data_,
                                                 buffer_[c].d_first_px_,
                                                 buffer_[c].d_cur_px_,
                                                 buffer_[c].d_cur_alpha_beta_,
                                                 buffer_[c].d_cur_f_,
                                                 buffer_[c].d_cur_disparity_,
                                                 stream_[c]);
    }
  }

  // 02) Process Tracking results
  for(std::size_t c=0;c<cur_frames->size();++c) {
    tracked_features_num_[c] = 0;
    std::vector<std::size_t> remove_indices;
    if(tracks_[c].size() > 0) {
      // Just synchronize with the stream..
      CUDA_API_CALL(cudaStreamSynchronize(stream_[c]));
      // And check the results
      for(std::size_t i=0; i<tracks_[c].size(); ++i) {
        struct FeatureTrack & track = tracks_[c][i];
        std::size_t offset = track.buffer_id_*METADATA_ELEMENT_BYTES/4;
        // if out px_x is nan, then remove the track, otherwise, update its current location
        float x = buffer_[c].h_cur_px_[offset];
        if(std::isnan(x)) {
          // we didnt converge in the KLT tracker
          remove_indices.push_back(i);
          releaseBufferId(track.buffer_id_,c);
  #if FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS
          life_stat_.add(track.life_);
  #endif /* FEATURE_TRACKER_ENABLE_ADDITIONAL_STATISTICS */
        } else {
          ++track.life_;
          // Last Position
          float y = buffer_[c].h_cur_px_[offset+1];
          track.cur_pos_[0] = x;
          track.cur_pos_[1] = y;
          // New disparity
          track.cur_disparity_ = buffer_[c].h_cur_disparity_[offset];
          // Update reference frame?
          if(options_.klt_template_is_first_observation == false) {
            // store shared pointer, so the frame is not freed
            track.template_frame_ = cur_base_frames[c];
            // update GPU template position
            buffer_[c].h_template_px_[offset] = x;
            buffer_[c].h_template_px_[offset+1] = y;
          }
          ++tracked_features_num_[c];
          addFeature(cur_base_frames[c],track);
        }
      }
    }
#if VERBOSE_TRACKING
    std::cout << " Tracked features (cam_id=" << c << "): " << tracked_features_num_[c] << std::endl;
#endif /* VERBOSE_TRACKING */
  // 02) Remove terminated tracks
#if VERBOSE_TRACKING
    std::cout << " Tracked features lost (cam_id=" << c << "): " << remove_indices.size() << std::endl;
#endif /* VERBOSE_TRACKING */
    removeTracks(remove_indices,c);
  }

  // 03) Detect new features (if necessary)
  for(std::size_t c=0;c<cur_frames->size();++c) {
    detected_features_num_[c] = 0;
    if(tracked_features_num_[c] < options_.min_tracks_to_detect_new_features) {
      OccupancyGrid2D & detector_grid = detector_[c]->getGrid();
      detector_grid.reset();
      // should we clear all tracks before detection?
      if(options_.reset_before_detection) {
        // yes, get rid of all tracked features
        tracked_features_num_[c] = 0;
        // free the buffers first
        for(auto track : tracks_[c]) {
          releaseBufferId(track.buffer_id_,c);
        }
        tracks_[c].clear();
        // discard all previously added features from tracking
        cur_base_frames[c]->num_features_ = 0;
      } else {
        // no, use the tracked features, and propagate the findings to the
        // detector
        for(std::size_t i=0;i<cur_base_frames[c]->num_features_;++i) {
          const Eigen::Vector2d & pos_2d = cur_base_frames[c]->px_vec_.col(i);
          detector_grid.setOccupied(pos_2d[0],pos_2d[1]);
        }
      }

      /*
       * Note to future self:
       * pre-occupied cells will be isEmpty(cell_index) = false
       */
      detector_[c]->detect(cur_base_frames[c].get()->pyramid_,
                           [&](const std::size_t & grid_cell_cnt,
                               const float * h_pos,
                               const float * h_score,
                               const int   * h_level) {
          // Do we use every detected feature?
          if(options_.use_best_n_features == -1) {
            // Yes, we are using every feature
            for(std::size_t cell_index=0;cell_index<detector_grid.size();++cell_index) {
              if(detector_grid.isEmpty(cell_index) && h_score[cell_index] > 0.0f) {
                // Add new feature track
                int track_index = addTrack(cur_base_frames[c],
                                           h_pos[cell_index*2],
                                           h_pos[cell_index*2+1],
                                           h_level[cell_index],
                                           h_score[cell_index],
                                           c);
                // Add tracked point to output
                addFeature(cur_base_frames[c],track_index,c);
                ++detected_features_num_[c];
              }
            }
          } else {
            // No, we are only using the best N features according to their score
            std::vector<std::size_t> idx(grid_cell_cnt);
            std::iota(idx.begin(), idx.end(), 0u);
            std::sort(idx.begin(), idx.end(), [&](std::size_t i1, std::size_t i2) {
              return h_score[i1] > h_score[i2];
            });
            std::size_t feature_limit = (std::size_t)std::max(0,options_.use_best_n_features - (int)tracked_features_num_[c]);
            for(std::size_t i=0;
                i<detector_grid.size() && detected_features_num_[c] < feature_limit;
                ++i) {
              std::size_t cell_index = idx[i];
              if(detector_grid.isEmpty(cell_index) && h_score[cell_index] > 0.0f) {
                // Add new feature track
                int track_index = addTrack(cur_base_frames[c],
                                           h_pos[cell_index*2],
                                           h_pos[cell_index*2+1],
                                           h_level[cell_index],
                                           h_score[cell_index],
                                           c);
                // Add tracked point to output
                addFeature(cur_base_frames[c],track_index,c);
                ++detected_features_num_[c];
              }
            }
          }
      });
    }
#if VERBOSE_TRACKING
    std::cout << " Detected features (cam_id=" << c << "): " << detected_features_num_[c] << std::endl;
#endif /* VERBOSE_TRACKING */

    // 04) Precompute patches & Hessians
    if(options_.klt_template_is_first_observation) {
      // only precompute the newly detected features
      updateTracks(detected_features_num_[c],cur_pyramids[c],c);
    } else {
      // precompute all current feature tracks
      updateTracks((detected_features_num_[c] + tracked_features_num_[c]),cur_pyramids[c],c);
    }
  }
  // 04) Do the accumulation for statistics
  total_tracked_features_num = std::accumulate(tracked_features_num_.begin(),
                                               tracked_features_num_.end(),
                                               0);
  total_detected_features_num = std::accumulate(detected_features_num_.begin(),
                                                detected_features_num_.end(),
                                                0);
}

void FeatureTrackerGPU::updateTracks(const std::size_t & last_n,
                                     const image_pyramid_descriptor_t & pyramid_description,
                                     const std::size_t & camera_id) {
  if(last_n == 0) {
    return;
  }

  std::size_t offset = tracks_[camera_id].size() - last_n;
  // initialize the indirection layer with used buffer ids
  std::size_t indir_id = 0;
  for(auto it=tracks_[camera_id].begin() + offset; it < tracks_[camera_id].end(); ++it) {
    buffer_[camera_id].h_indir_data_[indir_id++] = it->buffer_id_;
  }

  // start a kernel for initializing the templates and the inverse hessians
  feature_tracker_cuda_tools::update_tracks(last_n,
                                            options_.affine_est_offset,
                                            options_.affine_est_gain,
                                            options_.klt_min_level,
                                            options_.klt_max_level,
                                            pyramid_description,
                                            pyramid_patch_sizes_,
                                            buffer_[camera_id].d_indir_data_,
                                            buffer_[camera_id].d_template_px_,
                                            buffer_[camera_id].d_patch_data_,
                                            buffer_[camera_id].d_hessian_data_,
                                            stream_[camera_id]);
}

int FeatureTrackerGPU::addTrack(std::shared_ptr<Frame> & first_frame,
                                 const float & first_x,
                                 const float & first_y,
                                 const int & first_level,
                                 const float & first_score,
                                 const std::size_t & camera_id) {
  // Acquire free buffer id
  int buffer_id = acquireBufferId(camera_id);
  // Acquire new track id
  int track_id = Point::getNewId();
  // Create new feature track struct
  tracks_[camera_id].emplace_back(first_frame,
                                  first_x,
                                  first_y,
                                  first_level,
                                  first_score,
                                  track_id,
                                  buffer_id);
  // Update buffer elements
  std::size_t offset = buffer_id*METADATA_ELEMENT_BYTES/4;
  // Update ref_px   = {first_x,first_y}
  buffer_[camera_id].h_template_px_[offset] = first_x;
  buffer_[camera_id].h_template_px_[offset+1] = first_y;
  // Update first_px = {first_x,first_y}
  buffer_[camera_id].h_first_px_[offset] = first_x;
  buffer_[camera_id].h_first_px_[offset+1] = first_y;
  // Update search_px = {first_x,first_y}
  buffer_[camera_id].h_cur_px_[offset] = first_x;
  buffer_[camera_id].h_cur_px_[offset+1] = first_y;
  // Update alpha-beta = {0.0,0.0}
  buffer_[camera_id].h_cur_alpha_beta_[offset] = 0.0f;
  buffer_[camera_id].h_cur_alpha_beta_[offset+1] = 0.0f;
  return tracks_[camera_id].size()-1;
}

void FeatureTrackerGPU::setDetectorGPU(std::shared_ptr<DetectorBaseGPU> & detector,
                                       const std::size_t & camera_id) {
  // Free previous storage buffers if any
  freeStorage(camera_id);
  // Get detector, stream
  detector_[camera_id] = detector;
  stream_[camera_id] = detector->getStream();
  /*
   * Note to future self:
   * as at this point we already know the grid size, allocate the maximum number of
   * features to be tracked
   */
  /*
   * Note to future self:
   * 3 parts:
   * Layer of indirection: Array of integers, that point to AoS blocks [mapped memory]
   * Metadata elements: Array of structs [mapped memory]
   * Patchdata: array of structs [device memory]
   *
   * ----------------------------------------------------------------
   * Layer of indirection: consecutively placed ids point to buffers
   * int: 4 bytes
   * ----------------------------------------------------------------
   * Metadata: -> data stays in places
   * (16 + 8 + 8 + 8 + 8 + 4) = 52 bytes + 12 padding bytes = 64 bytes
   *
   * -> this could be either AoS or SoA
   * for the GPU-> since each warp uses exactly 1 parameter, it doesnt matter
   * for the CPU-> I think since the elements are closer, and we access 1 element, and then
   *               we jump to the next one, I'd assume AoS is faster
   *
   * [current]  search_bearing_vector: float (4x 4 bytes) [updated by the kernel]
   * [template] ref_px: float (2x 4 bytes) -> for patch preloading (can be the first or the current)
   * [first]    first_px: float (2x 4 bytes) -> for disparity calculation
   * [current]  search_px: float (2x 4 bytes) [updated by the kernel]
   * [current]  search_alpha_beta: float (2x 4 bytes) [updated by the kernel]
   * [current]  disparity: float (1x 4 bytes) [updated by the kernel]
   * ----------------------------------------------------------------
   * Patch data:
   * [template] patch level max_level
   * [template] patch level max_level-1
   * ...
   * [template] patch level min_level
   * ----------------------------------------------------------------
   * Hessian:
   * [template] inverse hessian level max_level
   * [template] inverse hessian level max_level-1
   * ...
   * [template] inverse hessian level min_level
   */
  std::size_t max_detected_ftr_count = 
    (options_.use_best_n_features==-1)?detector_[camera_id]->getGrid().size():options_.use_best_n_features;
  /*
   * Note to future self:
   * worst case scenario: all tracked features group in 1 cell, and their count drops below
   * min_tracks_to_detect_new_features, and then we detect (max_detected_ftr_count-1)
   */
  max_ftr_count_ = (options_.min_tracks_to_detect_new_features-1) + (max_detected_ftr_count-1);

  // 00) Create available indices for buffer bins
  // Note: the elements should be in decreasing order
  initBufferIds(camera_id);

  // 01) Allocate indirection layer
  CUDA_API_CALL(cudaHostAlloc((void**)&buffer_[camera_id].h_indir_data_,(max_ftr_count_ * sizeof(int)),cudaHostAllocMapped));
  CUDA_API_CALL(cudaHostGetDevicePointer((void**)&buffer_[camera_id].d_indir_data_,buffer_[camera_id].h_indir_data_,0));

  // 02) Metadata
  std::size_t metadata_bytes = (max_ftr_count_ * METADATA_ELEMENT_BYTES);
  // allocate mapped memory (as seen during feature alignment, it is going to be
  // faster, as we mainly only access 1 address once and then write out the result
  CUDA_API_CALL(cudaHostAlloc((void**)&buffer_[camera_id].h_metadata_,metadata_bytes,cudaHostAllocMapped));
  CUDA_API_CALL(cudaHostGetDevicePointer((void**)&buffer_[camera_id].d_metadata_,buffer_[camera_id].h_metadata_,0));

  // 03) Patchdata
  std::size_t patchdata_element_bytes = sizeof(FEATURE_TRACKER_REFERENCE_PATCH_TYPE)*pyramid_patch_sizes_.max_area*pyramid_levels_;
  std::size_t patchdata_bytes = patchdata_element_bytes * max_ftr_count_;
  // device memory, as the CPU doesnt need to access it
  CUDA_API_CALL(cudaMalloc((void**)&buffer_[camera_id].d_patch_data_,patchdata_bytes));

  // 04) Hessian data
  std::size_t hessian_element_bytes = sizeof(float)*10*pyramid_levels_;
  std::size_t hessian_bytes = hessian_element_bytes * max_ftr_count_;
  // device memory, as the CPU doesnt need to access it
  CUDA_API_CALL(cudaMalloc((void**)&buffer_[camera_id].d_hessian_data_,hessian_bytes));

  // Adjust device pointers
  unsigned char * d_ptr = buffer_[camera_id].d_metadata_;
  unsigned char * h_ptr = buffer_[camera_id].h_metadata_;
  // Bearing vector
  buffer_[camera_id].d_cur_f_ = (float4*)d_ptr; // 16-byte aligned
  buffer_[camera_id].h_cur_f_ = (float*)h_ptr;
  d_ptr += sizeof(float)*4;
  h_ptr += sizeof(float)*4;
  // Template position
  buffer_[camera_id].d_template_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_template_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // First position
  buffer_[camera_id].d_first_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_first_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current position
  buffer_[camera_id].d_cur_px_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_px_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current alpha-beta
  buffer_[camera_id].d_cur_alpha_beta_ = (float2*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_alpha_beta_ = (float*)h_ptr;
  d_ptr += sizeof(float)*2;
  h_ptr += sizeof(float)*2;
  // Current disparity
  buffer_[camera_id].d_cur_disparity_ = (float*)d_ptr; // 8-byte aligned
  buffer_[camera_id].h_cur_disparity_ = (float*)h_ptr;
}

void FeatureTrackerGPU::freeStorage(const std::size_t & camera_id) {
  // layer of indirection (mapped memory)
  if(buffer_[camera_id].h_indir_data_ != nullptr) {
    CUDA_API_CALL(cudaFreeHost(buffer_[camera_id].h_indir_data_));
    buffer_[camera_id].h_indir_data_ = nullptr;
  }
  // metadata (mapped memory)
  if(buffer_[camera_id].h_metadata_ != nullptr) {
    CUDA_API_CALL(cudaFreeHost(buffer_[camera_id].h_metadata_));
    buffer_[camera_id].h_metadata_ = nullptr;
  }
  // patch data
  if(buffer_[camera_id].d_patch_data_ != nullptr) {
    CUDA_API_CALL(cudaFree(buffer_[camera_id].d_patch_data_));
    buffer_[camera_id].d_patch_data_ = nullptr;
  }
  // hessian data (inverse)
  if(buffer_[camera_id].d_hessian_data_ != nullptr) {
    CUDA_API_CALL(cudaFree(buffer_[camera_id].d_hessian_data_));
    buffer_[camera_id].d_hessian_data_ = nullptr;
  }
}

void FeatureTrackerGPU::initBufferIds(const std::size_t & camera_id) {
  buffer_[camera_id].available_indices_.reserve(max_ftr_count_);
  for(std::size_t bin_id=0;bin_id<max_ftr_count_;++bin_id) {
    buffer_[camera_id].available_indices_.push_back(max_ftr_count_-bin_id-1);
  }
}

std::size_t FeatureTrackerGPU::acquireBufferId(const std::size_t & camera_id) {
  assert(buffer_[camera_id].available_indices_.size() > 0);
  std::size_t last_element = buffer_[camera_id].available_indices_.back();
  buffer_[camera_id].available_indices_.pop_back();
  return last_element;
}

void FeatureTrackerGPU::releaseBufferId(const std::size_t & id,
                                        const std::size_t & camera_id) {
  buffer_[camera_id].available_indices_.push_back(id);
}

void FeatureTrackerGPU::reset(void) {
  // release buffer ids
  for(std::size_t c=0;c<tracks_.size();++c) {
    for(const struct FeatureTrack & track : tracks_[c]) {
      releaseBufferId(track.buffer_id_,c);
    }
  }
  // clear the parent member variables
  FeatureTrackerBase::reset();
}

} // namespace vilib
