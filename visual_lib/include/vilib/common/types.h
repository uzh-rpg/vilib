/*
 * Common types
 * types.h
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