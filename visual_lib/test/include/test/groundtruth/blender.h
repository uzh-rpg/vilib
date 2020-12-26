/*
 * Blender functionalities and frame storage
 * blender.h
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

#pragma once

#include <assert.h>
#include <Eigen/Dense>

class Blender {
public:
  enum class Scenery {
    /* simple sequence for Sparse Image Alignment, and Feature Alignment */
    Hut=0,
    /* longer sequence for Feature Tracking */
    HutLong,
    /* simple sequence for Sparse Stereo matching */
    HutStereo
  };
  struct BlenderFrame {
    std::string image_2d_path_;
    std::string image_depth_path_;
    Eigen::Matrix4d T_W_C_;
    Eigen::Matrix4d T_C_W_;
  };

  Blender(Scenery scenery);
  ~Blender(void) = default;

  inline const struct BlenderFrame & getFrame(std::size_t idx) const {
    assert(frames_.size() > idx);
    return frames_[idx];
  }

  inline std::size_t getFrameCount(void) const {
    return frames_.size();
  }

  inline const Eigen::Vector4d & getIntrinsicParameters(void) const {
    return camera_intr_;
  }

  inline const Eigen::Matrix4d & get_T_B_C(void) const {
    return T_B_C_;
  }

  inline const Eigen::Matrix4d & get_T_C_B(void) const {
    return T_C_B_;
  }

  inline const Eigen::MatrixXd get_M(void) const {
    /*
     * Note to future self:
     * M = K * P
     */
    Eigen::MatrixXd P(3,4);
    Eigen::Matrix3d K;
    P << 1,0,0,0,
         0,1,0,0,
         0,0,1,0;
    K << camera_intr_(0), 0.0, camera_intr_(2),
         0.0, camera_intr_(1), camera_intr_(3),
         0.0, 0.0, 1.0;
    return (K*P);
  }

private:
  std::vector<struct BlenderFrame,
              Eigen::aligned_allocator<struct BlenderFrame>> frames_;
  Eigen::Vector4d camera_intr_;
  Eigen::Matrix4d T_B_C_;
  Eigen::Matrix4d T_C_B_;
};
