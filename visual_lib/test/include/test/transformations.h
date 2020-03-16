/*
 * Geometric transformations within the test suite
 * transformations.h
 */

#pragma once

#include <Eigen/Dense>

Eigen::Matrix4d get_T_W_C_from_Blender(const double * pose);
Eigen::Matrix4d get_T_inverse(const Eigen::Matrix4d & T);
