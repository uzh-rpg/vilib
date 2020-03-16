/*
 * transformations.cpp
 * Geometric transformations within the test suite
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
