/*
 * Common types
 * types.h
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