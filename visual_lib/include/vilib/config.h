/*
 * General configuration file
 * config.h
 */

#pragma once

#include "vilib/storage/subframe.h"

namespace vilib {

#define IMAGE_PYRAMID_MEMORY_TYPE               Subframe::MemoryType::PITCHED_DEVICE_MEMORY
#define IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM    10
#define MAX_IMAGE_PYRAMID_LEVELS                5

} // namespace vilib
