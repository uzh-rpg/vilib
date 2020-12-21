/*
 * Geometric transformations within the test suite
 * transformations.cpp
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

#include "test/transformations.h"

Eigen::Matrix4d get_T_W_C_from_Blender(const double * pose) {
  const double & pos_x = pose[0];
  const double & pos_y = pose[1];
  const double & pos_z = pose[2];
  const double & q_w   = pose[3];
  const double & q_x   = pose[4];
  const double & q_y   = pose[5];
  const double & q_z   = pose[6];
  // Transformation from World to BlenderCam: T_W_BC
  /*
   * Note to future self:
   * we need to normalize the quaternion because of read-off errors
   */
  Eigen::Matrix4d T_W_BC = Eigen::Matrix4d::Identity();
  Eigen::Quaterniond q(q_w,q_x,q_y,q_z);
  q.normalize();
  T_W_BC.block<3,3>(0,0) = q.toRotationMatrix();
  T_W_BC.block<3,1>(0,3) << pos_x, pos_y, pos_z;

  // Transformation from BlenderCam to ComputerVisionCam: T_BC_C
  Eigen::Matrix4d T_BC_C;
  T_BC_C << 1.0,  0.0,  0.0, 0.0,
            0.0, -1.0,  0.0, 0.0,
            0.0,  0.0, -1.0, 0.0,
            0.0,  0.0,  0.0, 1.0;
  return (T_W_BC * T_BC_C);
}

Eigen::Matrix4d get_T_inverse(const Eigen::Matrix4d & T) {
  /* 
   * Note: we make sure in this implementation that the rotation matrix stays
   * orthonormal
   */
  Eigen::Matrix4d T_inv;
  T_inv.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
  T_inv.block<3,1>(0,3) = -1* T_inv.block<3,3>(0,0) * T.block<3,1>(0,3);
  T_inv.block<1,4>(3,0) << 0.0, 0.0, 0.0, 1.0;
  return T_inv;
}
