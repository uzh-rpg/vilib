/*
 * Base class for GPU feature detectors
 * config.h
 */

#pragma once

namespace vilib {

// Non-maximum suppression size
#define DETECTOR_BASE_NMS_SIZE                          3

#if (DETECTOR_BASE_NMS_SIZE%2 == 0)
#error "Error: the non-maximum suppression size should be an odd number"
#endif /* (DETECTOR_BASE_NMS_SIZE%2 == 0) */

} // namespace vilib
