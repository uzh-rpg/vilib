/*
 * Blender functionalities and frame storage
 * blender.cpp
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "test/groundtruth/blender.h"
#include "test/transformations.h"

#define DISPLAY_CAMERA_PARAMS       0
#define DISPLAY_FRAME_PARAMS        0

Blender::Blender(Scenery scenery) {
  /*
   * Create a Blender camera
   * Note to future self:
   * The 2 current render engines of blender (2.79b),
   * "Blender Render" & "Cycles Render" create slightly different depth maps
   *
   * In "Cycles Render" the depth map distances are the Euclidian distances of the
   * 3D points and the Computer vision camera frame's origin.
   */
  // camera_intr [ fx, fy, cx, cy ]

  // Create camera parameters from JSON
  std::string json_path;
  switch(scenery) {
    case Scenery::Hut:
      json_path = "test/images/scenery/hut.json";
      break;
    case Scenery::HutLong:
      json_path = "test/images/scenery/hut_long.json";
      break;
    case Scenery::HutStereo:
      json_path = "test/images/scenery/hut_stereo.json";
      break;
    default:
      assert(0);
      break;
  }
  // Blender camera parameters
  try {
    // Open JSON file
    nlohmann::json json;
    std::ifstream json_stream(json_path);
    json_stream >> json;

    // Camera properties
    nlohmann::json camera = json["camera"];
    double fx = camera["fx"];
    double fy = camera["fy"];
    double cx = camera["cx"];
    double cy = camera["cy"];
#if DISPLAY_CAMERA_PARAMS
    std::cout << "Fx: " << fx << std::endl;
    std::cout << "Fy: " << fy << std::endl;
    std::cout << "Cx: " << cx << std::endl;
    std::cout << "Cy: " << cy << std::endl;
#endif /* DISPLAY_CAMERA_PARAMS */

    camera_intr_ << fx, fy, cx, cy;
    T_B_C_ << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
             0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
             0.0, 0.0, 0.0, 1.0;
    T_C_B_ = get_T_inverse(T_B_C_);

    // Frames, paths, poses
    std::vector<std::string> image_2d_paths = json["image_2d_paths"];
    std::vector<std::string> image_depth_paths = json["image_depth_paths"];
    std::vector<nlohmann::json> camera_poses = json["camera_poses"];
    // check if the number of data is the same
    if(image_2d_paths.size() != image_depth_paths.size() ||
       image_depth_paths.size() != camera_poses.size()) {
      std::cout << "Error: the number of datapoints is not equal" << std::endl;
      return;
    }
    for(std::size_t i=0;i<image_2d_paths.size();++i) {
      struct BlenderFrame bf;
      /*
       * std::string image_2d_path_;
       * std::string image_depth_path_;
       * Eigen::Matrix4d T_W_C_;
       * Eigen::Matrix4d T_C_W_;
       */
      bf.image_2d_path_ = "test/images/scenery/";
      bf.image_2d_path_ += image_2d_paths[i];
      bf.image_depth_path_ = "test/images/scenery/";
      bf.image_depth_path_ += image_depth_paths[i];
      std::vector<double> pose = camera_poses[i];
      bf.T_W_C_ = get_T_W_C_from_Blender(pose.data());
      bf.T_C_W_ = get_T_inverse(bf.T_W_C_);
      frames_.push_back(bf);
#if DISPLAY_FRAME_PARAMS
      std::cout << "Frame " << (i+1) << std::endl;
      std::cout << " " << image_2d_paths[i] << std::endl;
      std::cout << " " << image_depth_paths[i] << std::endl;
      std::cout << " " << camera_poses[i] << std::endl;
      std::cout << "---" << std::endl;
#endif /* DISPLAY_FRAME_PARAMS */
    }
  }
  catch(std::exception & e) {
    std::cout << "Error: invalid JSON format." << std::endl;
    std::cout << "Reason: " << e.what() << std::endl;
    return;
  }
}
