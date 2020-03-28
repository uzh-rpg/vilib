/*
 * Common types
 * types.h
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

#pragma once

#include "vilib/config.h"

namespace vilib {

/*
 * Image pramids
 * Pyramid descriptor: only holds information about the pyramid structure
 * Image pyramid descriptor: contains the pyramid descriptor + pointers for the
 *                           images
 */
struct pyramid_descriptor {
  int w[MAX_IMAGE_PYRAMID_LEVELS]; // image width  [pixel units]
  int h[MAX_IMAGE_PYRAMID_LEVELS]; // image height [pixel units]
  int p[MAX_IMAGE_PYRAMID_LEVELS]; // image pitch  [byte units]
  int l;                           // actual number of utilized levels
};
typedef struct pyramid_descriptor pyramid_descriptor_t;

struct image_pyramid_descriptor {
  pyramid_descriptor_t desc;
  unsigned char * __restrict__ data[MAX_IMAGE_PYRAMID_LEVELS];
};
typedef struct image_pyramid_descriptor image_pyramid_descriptor_t;

/*
 * Pyramid patch descriptor
 * Since we use these direct matchers, we use patch sizes to do the
 * inverse-compositional Lukas-Kanade algorithm.
 */
struct pyramid_patch_descriptor {
  int wh[MAX_IMAGE_PYRAMID_LEVELS]; // image width and height (symmetric) [pixel units]
  int max_area;                     // max area (w_pixels * h_pixels) per feature patch (including borders)
};
typedef struct pyramid_patch_descriptor pyramid_patch_descriptor_t;

} // namespace vilib